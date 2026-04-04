"""Base HTTP client with retry logic and rate limiting."""

from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pipeline.utils.logging import get_logger
from pipeline.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIError):
    """API rate limit exceeded."""

    pass


class AuthenticationError(APIError):
    """API authentication failed."""

    pass


class BaseAPIClient:
    """Base HTTP client with retry logic and rate limiting.

    All API clients should inherit from this class to get consistent
    behavior for rate limiting, retries, and error handling.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        requests_per_second: float = 10.0,
        timeout: float = 30.0,
        user_agent: Optional[str] = None,
    ):
        """Initialize the base API client.

        Args:
            base_url: Base URL for the API
            api_key: API key for authentication (if required)
            requests_per_second: Maximum request rate
            timeout: Request timeout in seconds
            user_agent: Custom User-Agent header
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        self.rate_limiter = RateLimiter(requests_per_second=requests_per_second)

        headers = {}
        if user_agent:
            headers["User-Agent"] = user_agent

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=headers,
            follow_redirects=True,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "BaseAPIClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint.

        Args:
            endpoint: API endpoint path

        Returns:
            Full URL
        """
        if endpoint.startswith("http"):
            return endpoint
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error HTTP responses.

        Args:
            response: HTTP response object

        Raises:
            RateLimitError: If rate limited (429)
            AuthenticationError: If authentication failed (401/403)
            APIError: For other HTTP errors
        """
        if response.status_code == 429:
            raise RateLimitError(
                "Rate limit exceeded",
                status_code=response.status_code,
            )

        if response.status_code in (401, 403):
            raise AuthenticationError(
                f"Authentication failed: {response.text}",
                status_code=response.status_code,
            )

        if response.status_code >= 400:
            raise APIError(
                f"API error: {response.status_code} - {response.text}",
                status_code=response.status_code,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((httpx.HTTPError, RateLimitError)),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response:
        """Make an HTTP request with rate limiting and retries.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            json: JSON body for POST/PUT requests
            headers: Additional headers

        Returns:
            HTTP response object

        Raises:
            APIError: If the request fails after retries
        """
        await self.rate_limiter.acquire()

        url = self._build_url(endpoint)

        logger.debug(
            "Making API request",
            method=method,
            url=url,
            params=params,
        )

        try:
            response = await self._client.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=headers,
            )

            self._handle_error_response(response)
            return response

        except httpx.HTTPError as e:
            logger.error(
                "HTTP request failed",
                method=method,
                url=url,
                error=str(e),
            )
            raise

    async def get(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response:
        """Make a GET request.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            headers: Additional headers

        Returns:
            HTTP response object
        """
        return await self._request("GET", endpoint, params=params, headers=headers)

    async def get_json(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Any:
        """Make a GET request and return JSON response.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            headers: Additional headers

        Returns:
            Parsed JSON response
        """
        response = await self.get(endpoint, params=params, headers=headers)
        return response.json()

    async def post(
        self,
        endpoint: str,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response:
        """Make a POST request.

        Args:
            endpoint: API endpoint path
            json: JSON body
            params: Query parameters
            headers: Additional headers

        Returns:
            HTTP response object
        """
        return await self._request(
            "POST", endpoint, params=params, json=json, headers=headers
        )
