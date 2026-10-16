import cv2

def praproses(citra, clip=2.0, grid=8):
    """Rangkaian praproses tim. Parameter WAJIB sama untuk seluruh data."""
    L, A, B = cv2.split(cv2.cvtColor(citra, cv2.COLOR_BGR2LAB))
    L = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(L)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
