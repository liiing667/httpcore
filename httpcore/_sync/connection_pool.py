from __future__ import annotations

import ssl
import sys
import time
import types
import typing

from .._backends.sync import SyncBackend
from .._backends.base import SOCKET_OPTION, NetworkBackend
from .._exceptions import (
    ConnectionNotAvailable,
    NetworkError,
    RemoteProtocolError,
    UnsupportedProtocol,
)
from .._models import Origin, Proxy, Request, Response
from .._synchronization import Event, ShieldCancellation, ThreadLock
from .connection import HTTPConnection
from .interfaces import ConnectionInterface, RequestInterface

# HTTP methods that are safe to retry on a fresh connection if a
# reused connection turns out to have been silently closed.
IDEMPOTENT_METHODS = frozenset(
    {b"GET", b"HEAD", b"PUT", b"DELETE", b"OPTIONS", b"TRACE"}
)


class PoolRequest:
    def __init__(self, request: Request) -> None:
        self.request = request
        self.connection: ConnectionInterface | None = None
        # Set when the assigned connection was reused from the pool,
        # rather than being newly established.
        self.reused_connection = False
        # Set once a request has been retried after failing on a stale
        # reused connection, so that it is not retried more than once.
        self.stale_retry_used = False
        # Set when a request must be assigned a freshly established
        # connection, rather than reusing an existing one.
        self.force_new_connection = False
        self._connection_acquired = Event()

    def assign_to_connection(
        self, connection: ConnectionInterface | None, reused: bool = False
    ) -> None:
        self.connection = connection
        self.reused_connection = reused
        self._connection_acquired.set()

    def clear_connection(self) -> None:
        self.connection = None
        self.reused_connection = False
        self._connection_acquired = Event()

    def wait_for_connection(
        self, timeout: float | None = None
    ) -> ConnectionInterface:
        if self.connection is None:
            self._connection_acquired.wait(timeout=timeout)
        assert self.connection is not None
        return self.connection

    def is_queued(self) -> bool:
        return self.connection is None


class ConnectionPool(RequestInterface):
    """
    A connection pool for making HTTP requests.
    """

    def __init__(
        self,
        ssl_context: ssl.SSLContext | None = None,
        proxy: Proxy | None = None,
        max_connections: int | None = 10,
        max_keepalive_connections: int | None = None,
        keepalive_expiry: float | None = None,
        max_connection_age: float | None = None,
        retry_stale_connections: bool = True,
        http1: bool = True,
        http2: bool = False,
        retries: int = 0,
        local_address: str | None = None,
        uds: str | None = None,
        network_backend: NetworkBackend | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        """
        A connection pool for making HTTP requests.

        Parameters:
            ssl_context: An SSL context to use for verifying connections.
                If not specified, the default `httpcore.default_ssl_context()`
                will be used.
            max_connections: The maximum number of concurrent HTTP connections that
                the pool should allow. Any attempt to send a request on a pool that
                would exceed this amount will block until a connection is available.
            max_keepalive_connections: The maximum number of idle HTTP connections
                that will be maintained in the pool.
            keepalive_expiry: The duration in seconds that an idle HTTP connection
                may be maintained for before being expired from the pool.
            max_connection_age: The maximum duration in seconds that an HTTP
                connection may be reused for, counted from when it is added to
                the pool. An idle connection that exceeds this age is recycled
                instead of being reused. If `None`, connections do not expire
                by age. Unlike `keepalive_expiry` this timer is not reset while
                a connection is in the pool.
            retry_stale_connections: A boolean indicating if requests that fail
                on a connection reused from the pool should be retried once on a
                new connection. Only idempotent requests (GET, HEAD, PUT,
                DELETE, OPTIONS, TRACE) are retried. Defaults to `True`.
            http1: A boolean indicating if HTTP/1.1 requests should be supported
                by the connection pool. Defaults to True.
            http2: A boolean indicating if HTTP/2 requests should be supported by
                the connection pool. Defaults to False.
            retries: The maximum number of retries when trying to establish a
                connection.
            local_address: Local address to connect from. Can also be used to connect
                using a particular address family. Using `local_address="0.0.0.0"`
                will connect using an `AF_INET` address (IPv4), while using
                `local_address="::"` will connect using an `AF_INET6` address (IPv6).
            uds: Path to a Unix Domain Socket to use instead of TCP sockets.
            network_backend: A backend instance to use for handling network I/O.
            socket_options: Socket options that have to be included
             in the TCP socket when the connection was established.
        """
        self._ssl_context = ssl_context
        self._proxy = proxy
        self._max_connections = (
            sys.maxsize if max_connections is None else max_connections
        )
        self._max_keepalive_connections = (
            sys.maxsize
            if max_keepalive_connections is None
            else max_keepalive_connections
        )
        self._max_keepalive_connections = min(
            self._max_connections, self._max_keepalive_connections
        )

        self._keepalive_expiry = keepalive_expiry
        self._max_connection_age = max_connection_age
        self._retry_stale_connections = retry_stale_connections
        self._http1 = http1
        self._http2 = http2
        self._retries = retries
        self._local_address = local_address
        self._uds = uds

        self._network_backend = (
            SyncBackend() if network_backend is None else network_backend
        )
        self._socket_options = socket_options

        # The mutable state on a connection pool is the queue of incoming requests,
        # and the set of connections that are servicing those requests.
        self._connections: list[ConnectionInterface] = []
        self._requests: list[PoolRequest] = []
        # The time at which each connection was added to the pool.
        self._connection_birth: dict[ConnectionInterface, float] = {}

        # Counters exposed by `get_stats()`.
        self._stats = {"recycled": 0, "retried": 0, "idle_expired": 0}

        # We only mutate the state of the connection pool within an 'optional_thread_lock'
        # context. This holds a threading lock unless we're running in async mode,
        # in which case it is a no-op.
        self._optional_thread_lock = ThreadLock()

    def create_connection(self, origin: Origin) -> ConnectionInterface:
        if self._proxy is not None:
            if self._proxy.url.scheme in (b"socks5", b"socks5h"):
                from .socks_proxy import Socks5Connection

                return Socks5Connection(
                    proxy_origin=self._proxy.url.origin,
                    proxy_auth=self._proxy.auth,
                    remote_origin=origin,
                    ssl_context=self._ssl_context,
                    keepalive_expiry=self._keepalive_expiry,
                    http1=self._http1,
                    http2=self._http2,
                    network_backend=self._network_backend,
                )
            elif origin.scheme == b"http":
                from .http_proxy import ForwardHTTPConnection

                return ForwardHTTPConnection(
                    proxy_origin=self._proxy.url.origin,
                    proxy_headers=self._proxy.headers,
                    proxy_ssl_context=self._proxy.ssl_context,
                    remote_origin=origin,
                    keepalive_expiry=self._keepalive_expiry,
                    network_backend=self._network_backend,
                )
            from .http_proxy import TunnelHTTPConnection

            return TunnelHTTPConnection(
                proxy_origin=self._proxy.url.origin,
                proxy_headers=self._proxy.headers,
                proxy_ssl_context=self._proxy.ssl_context,
                remote_origin=origin,
                ssl_context=self._ssl_context,
                keepalive_expiry=self._keepalive_expiry,
                http1=self._http1,
                http2=self._http2,
                network_backend=self._network_backend,
            )

        return HTTPConnection(
            origin=origin,
            ssl_context=self._ssl_context,
            keepalive_expiry=self._keepalive_expiry,
            http1=self._http1,
            http2=self._http2,
            retries=self._retries,
            local_address=self._local_address,
            uds=self._uds,
            network_backend=self._network_backend,
            socket_options=self._socket_options,
        )

    @property
    def connections(self) -> list[ConnectionInterface]:
        """
        Return a list of the connections currently in the pool.

        For example:

        ```python
        >>> pool.connections
        [
            <HTTPConnection ['https://example.com:443', HTTP/1.1, ACTIVE, Request Count: 6]>,
            <HTTPConnection ['https://example.com:443', HTTP/1.1, IDLE, Request Count: 9]> ,
            <HTTPConnection ['http://example.com:80', HTTP/1.1, IDLE, Request Count: 1]>,
        ]
        ```
        """
        return list(self._connections)

    def handle_request(self, request: Request) -> Response:
        """
        Send an HTTP request, and return an HTTP response.

        This is the core implementation that is called into by `.request()` or `.stream()`.
        """
        scheme = request.url.scheme.decode()
        if scheme == "":
            raise UnsupportedProtocol(
                "Request URL is missing an 'http://' or 'https://' protocol."
            )
        if scheme not in ("http", "https", "ws", "wss"):
            raise UnsupportedProtocol(
                f"Request URL has an unsupported protocol '{scheme}://'."
            )

        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("pool", None)

        with self._optional_thread_lock:
            # Add the incoming request to our request queue.
            pool_request = PoolRequest(request)
            self._requests.append(pool_request)

        try:
            while True:
                with self._optional_thread_lock:
                    # Assign incoming requests to available connections,
                    # closing or creating new connections as required.
                    closing = self._assign_requests_to_connections()
                self._close_connections(closing)

                # Wait until this request has an assigned connection.
                connection = pool_request.wait_for_connection(timeout=timeout)

                try:
                    # Send the request on the assigned connection.
                    response = connection.handle_request(
                        pool_request.request
                    )
                except ConnectionNotAvailable:
                    # In some cases a connection may initially be available to
                    # handle a request, but then become unavailable.
                    #
                    # In this case we clear the connection and try again.
                    pool_request.clear_connection()
                except (NetworkError, RemoteProtocolError) as exc:
                    # A connection that was reused from the pool may have been
                    # silently closed by the remote host. Idempotent requests
                    # are retried once on a freshly established connection.
                    if not self._should_retry_stale_connection(pool_request):
                        raise exc from None
                    with self._optional_thread_lock:
                        if connection in self._connections:
                            self._remove_connection(connection)
                            self._stats["recycled"] += 1
                            closing = [connection]
                        else:
                            closing = []
                        self._stats["retried"] += 1
                        pool_request.clear_connection()
                        pool_request.stale_retry_used = True
                        pool_request.force_new_connection = True
                    self._close_connections(closing)
                else:
                    break  # pragma: nocover

        except BaseException as exc:
            with self._optional_thread_lock:
                # For any exception or cancellation we remove the request from
                # the queue, and then re-assign requests to connections.
                self._requests.remove(pool_request)
                closing = self._assign_requests_to_connections()

            self._close_connections(closing)
            raise exc from None

        # Return the response. Note that in this case we still have to manage
        # the point at which the response is closed.
        assert isinstance(response.stream, typing.Iterable)
        return Response(
            status=response.status,
            headers=response.headers,
            content=PoolByteStream(
                stream=response.stream, pool_request=pool_request, pool=self
            ),
            extensions=response.extensions,
        )

    def _should_retry_stale_connection(self, pool_request: PoolRequest) -> bool:
        return (
            self._retry_stale_connections
            and pool_request.reused_connection
            and not pool_request.stale_retry_used
            and pool_request.request.method in IDEMPOTENT_METHODS
        )

    def _remove_connection(self, connection: ConnectionInterface) -> None:
        self._connections.remove(connection)
        self._connection_birth.pop(connection, None)

    def _assign_requests_to_connections(self) -> list[ConnectionInterface]:
        """
        Manage the state of the connection pool, assigning incoming
        requests to connections as available.

        Called whenever a new request is added or removed from the pool.

        Any closing connections are returned, allowing the I/O for closing
        those connections to be handled seperately.
        """
        closing_connections = []
        now = time.monotonic()

        # First we handle cleaning up any connections that are closed,
        # have exceeded their maximum age, have expired their keep-alive,
        # or surplus idle connections.
        for connection in list(self._connections):
            if connection.is_closed():
                # log: "removing closed connection"
                self._remove_connection(connection)
            elif (
                connection.is_idle()
                and self._max_connection_age is not None
                and now - self._connection_birth.get(connection, now)
                > self._max_connection_age
            ):
                # log: "recycling connection past its maximum age"
                self._remove_connection(connection)
                closing_connections.append(connection)
                self._stats["recycled"] += 1
            elif connection.has_expired():
                # log: "closing expired connection"
                self._remove_connection(connection)
                closing_connections.append(connection)
                self._stats["idle_expired"] += 1
            elif (
                connection.is_idle()
                and sum(connection.is_idle() for connection in self._connections)
                > self._max_keepalive_connections
            ):
                # log: "closing idle connection"
                self._remove_connection(connection)
                closing_connections.append(connection)

        # Assign queued requests to connections.
        queued_requests = [request for request in self._requests if request.is_queued()]
        for pool_request in queued_requests:
            origin = pool_request.request.url.origin
            if pool_request.force_new_connection:
                available_connections = []
            else:
                available_connections = [
                    connection
                    for connection in self._connections
                    if connection.can_handle_request(origin)
                    and connection.is_available()
                ]
            idle_connections = [
                connection for connection in self._connections if connection.is_idle()
            ]

            # There are three cases for how we may be able to handle the request:
            #
            # 1. There is an existing connection that can handle the request.
            # 2. We can create a new connection to handle the request.
            # 3. We can close an idle connection and then create a new connection
            #    to handle the request.
            if available_connections:
                # log: "reusing existing connection"
                connection = available_connections[0]
                pool_request.assign_to_connection(
                    connection, reused=connection.is_idle()
                )
            elif len(self._connections) < self._max_connections:
                # log: "creating new connection"
                connection = self.create_connection(origin)
                self._connections.append(connection)
                self._connection_birth[connection] = now
                pool_request.assign_to_connection(connection)
            elif idle_connections:
                # log: "closing idle connection"
                connection = idle_connections[0]
                self._remove_connection(connection)
                closing_connections.append(connection)
                # log: "creating new connection"
                connection = self.create_connection(origin)
                self._connections.append(connection)
                self._connection_birth[connection] = now
                pool_request.assign_to_connection(connection)

        return closing_connections

    def _close_connections(self, closing: list[ConnectionInterface]) -> None:
        # Close connections which have been removed from the pool.
        with ShieldCancellation():
            for connection in closing:
                connection.close()

    def close(self) -> None:
        # Explicitly close the connection pool.
        # Clears all existing requests and connections.
        with self._optional_thread_lock:
            closing_connections = list(self._connections)
            self._connections = []
            self._connection_birth = {}
        self._close_connections(closing_connections)

    def get_stats(self) -> dict[str, int]:
        """
        Return a snapshot of the connection pool health counters.

        * `recycled`: Connections discarded by the pool health policy, either
          for exceeding `max_connection_age`, or for failing on reuse.
        * `retried`: Requests retried on a new connection after a reused
          connection failed.
        * `idle_expired`: Idle connections expired by `keepalive_expiry`, or
          closed by the remote host while idle.
        """
        with self._optional_thread_lock:
            return dict(self._stats)

    def __enter__(self) -> ConnectionPool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        with self._optional_thread_lock:
            request_is_queued = [request.is_queued() for request in self._requests]
            connection_is_idle = [
                connection.is_idle() for connection in self._connections
            ]

            num_active_requests = request_is_queued.count(False)
            num_queued_requests = request_is_queued.count(True)
            num_active_connections = connection_is_idle.count(False)
            num_idle_connections = connection_is_idle.count(True)

        requests_info = (
            f"Requests: {num_active_requests} active, {num_queued_requests} queued"
        )
        connection_info = (
            f"Connections: {num_active_connections} active, {num_idle_connections} idle"
        )

        return f"<{class_name} [{requests_info} | {connection_info}]>"


class PoolByteStream:
    def __init__(
        self,
        stream: typing.Iterable[bytes],
        pool_request: PoolRequest,
        pool: ConnectionPool,
    ) -> None:
        self._stream = stream
        self._pool_request = pool_request
        self._pool = pool
        self._closed = False

    def __iter__(self) -> typing.Iterator[bytes]:
        try:
            for part in self._stream:
                yield part
        except BaseException as exc:
            self.close()
            raise exc from None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with ShieldCancellation():
                if hasattr(self._stream, "close"):
                    self._stream.close()

            with self._pool._optional_thread_lock:
                self._pool._requests.remove(self._pool_request)
                closing = self._pool._assign_requests_to_connections()

            self._pool._close_connections(closing)
