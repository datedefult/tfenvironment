from typing import Any, Optional
from typing import Generic, TypeVar

from fastapi import status
from pydantic import BaseModel
from starlette.responses import JSONResponse

T = TypeVar("T")


class GeneralResponse(BaseModel, Generic[T]):
    code: int
    message: str
    data: T | None = None
    class Config:
        extra = 'allow'

def success_response(data: Optional[Any] = None, message: str = "Success", code: int = status.HTTP_200_OK,**kwargs):
    return GeneralResponse(code=code, message=message, data=data, **kwargs)


def error_response(message: str = "An error occurred", data: Optional[Any] = None,
        code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
    # raise HTTPException(
    #     status_code=code,
    #     detail=BaseResponse(code=code, message=message, data=data).model_dump()
    # )
    return JSONResponse(content=GeneralResponse(code=code, message=message, data=data).model_dump(), status_code=code)
