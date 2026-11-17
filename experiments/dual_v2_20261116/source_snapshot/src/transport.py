"""Classify exception chains into fixed labels; never serialize exception text."""
import errno
import socket
import ssl

CONNECTION_HINTS = {
    "dns_resolution_failed": "Name resolution failed for the endpoint or proxy; check DNS and configured host names.",
    "tls_certificate_error": "TLS certificate verification failed; check system time and the trusted CA configuration. Keep TLS verification enabled.",
    "tls_handshake_error": "TLS negotiation failed; check endpoint scheme, proxy compatibility and network interception.",
    "connection_refused": "A connection was refused by the endpoint or proxy; check the configured address, port and local proxy service.",
    "connection_reset": "The peer or an intermediary reset the connection; check endpoint availability, proxy and firewall.",
    "network_unreachable": "The endpoint or proxy network is unreachable; check routing, firewall and network connectivity.",
    "connection_failed": "Connection failed without a specific safe cause; use the unauthenticated connection check before retrying generation.",
}


def connection_reason(exc):
    """Inspect at most 12 linked exceptions without logging messages or URLs."""
    stack, seen, matches = [exc], set(), set()
    while stack and len(seen) < 12:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, socket.gaierror):
            matches.add("dns_resolution_failed")
        if isinstance(current, ssl.SSLCertVerificationError):
            matches.add("tls_certificate_error")
        elif isinstance(current, ssl.SSLError):
            matches.add("tls_handshake_error")
        codes = {getattr(current, "errno", None), getattr(current, "winerror", None)}
        if isinstance(current, ConnectionRefusedError) or codes & {errno.ECONNREFUSED, 10061}:
            matches.add("connection_refused")
        if isinstance(current, ConnectionResetError) or codes & {errno.ECONNRESET, 10054}:
            matches.add("connection_reset")
        if codes & {errno.ENETUNREACH, errno.EHOSTUNREACH, 10051, 10065}:
            matches.add("network_unreachable")
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                stack.append(linked)
    return next((key for key in CONNECTION_HINTS if key in matches), "connection_failed")
