class AppError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "app_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class NotFoundError(AppError):
    def __init__(self, message: str = "资源不存在"):
        super().__init__(message, status_code=404, code="not_found")


class ConflictError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=409, code="conflict")


class ValidationError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=422, code="validation_error")


class UpstreamError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=502, code="upstream_error")
