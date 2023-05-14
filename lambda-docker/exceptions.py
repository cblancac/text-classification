from abc import ABC
from typing import List


class EndpointError(ABC):
    def __init__(self, message: str, status_code: int):
        self.message = message
        self.status_code = status_code

    def response(self) -> dict:
        return {"message": self.message, "status_code": self.status_code}


class BadContentType(EndpointError):
    def __init__(self, content_types: List[str]):
        message = "Bad Content-Type expected: " + ", ".join(content_types)
        status_code = 415
        super().__init__(message, status_code)


class BadRequest(EndpointError):
    def __init__(self):
        message = "Wrong content body"
        status_code = 400
        super().__init__(message, status_code)


class ResourceNotFound(EndpointError):
    def __init__(self, resource: str) -> None:
        message = "Can't retrieve content from " + resource
        status_code = 404
        super().__init__(message, status_code)


class ResourceForbidden(EndpointError):
    def __init__(self, resource: str) -> None:
        message = "Can't access denied to " + resource
        status_code = 403
        super().__init__(message, status_code)