"""
自定义异常类
"""


class ValueValidException(Exception):
    """
    值验证错误
    """

    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        return self._msg


class ValueEmptyException(Exception):
    """
    空值异常
    """
    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        return self._msg
