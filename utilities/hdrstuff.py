import cv2 as cv
import numpy as np

# ffmpeg -s 1920x1080 -r 30 -pix_fmt yuv420p10le -i ./test/out/va_f1.yuv -t 5 -c:v libx265 -tag:v hvc1 -x265-params "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=full" -pix_fmt yuv420p10le -r 30 ./test/out/va_f1.yuv.mp4 -y
# https://stackoverflow.com/a/76050556
def RGB_to_YUV(RGB, gamut="bt2020", bits=10, video_range="full", formation="420"):
    if RGB.dtype == "uint8":
        RGB = RGB / 255.0
    height, width = RGB.shape[:2]

    if bits == 8:
        dtype = np.uint8
    else:
        dtype = np.uint16

    if gamut == "bt709":
        YCbCr = RGB709_to_YCbCr709(RGB)
    elif gamut == "bt2020":
        YCbCr = RGB2020_to_YCbCr2020(RGB)
    else:
        raise Exception("gamut param error!")

    Y = YCbCr[..., 0]
    Cb = YCbCr[..., 1]
    Cr = YCbCr[..., 2]

    if video_range == "limited":
        D_Y = np.clip(np.round(Y * 219 + 16), 16, 235).astype(dtype) * np.power(
            2, bits - 8
        )
        D_Cb = np.clip(np.round(Cb * 224 + 128), 16, 240).astype(dtype) * np.power(
            2, bits - 8
        )
        D_Cr = np.clip(np.round(Cr * 224 + 128), 16, 240).astype(dtype) * np.power(
            2, bits - 8
        )

    elif video_range == "full":
        D_Y = np.clip(np.round(Y * 255), 0, 255).astype(dtype) * np.power(2, bits - 8)
        D_Cb = np.clip(np.round(Cb * 254 + 128), 1, 255).astype(dtype) * np.power(
            2, bits - 8
        )
        D_Cr = np.clip(np.round(Cr * 254 + 128), 1, 255).astype(dtype) * np.power(
            2, bits - 8
        )

    else:
        raise Exception("param: video_range error!")

    y_size = height * width
    uv_size = height // 2 * width // 2
    frame_len = y_size * 3 // 2

    if formation == "420":
        U = cv.resize(D_Cb, None, None, 0.5, 0.5).flatten()
        V = cv.resize(D_Cr, None, None, 0.5, 0.5).flatten()

        yuv = np.empty(frame_len, dtype=dtype)
        yuv[:y_size] = D_Y.flatten()
        yuv[y_size : y_size + uv_size] = U
        yuv[y_size + uv_size :] = V
        return yuv

    elif formation == "444":
        Y = D_Y
        U = D_Cb
        V = D_Cr
        return cv.merge((Y, U, V))


def RGB709_to_YCbCr709(RGB):
    if RGB.dtype == "uint8":
        RGB = RGB / 255.0

    m_RGB709_to_YCbCr709 = np.array(
        [
            [0.21260000, 0.71520000, 0.07220000],
            [-0.11457211, -0.38542789, 0.50000000],
            [0.50000000, -0.45415291, -0.04584709],
        ]
    )

    return np.matmul(RGB, m_RGB709_to_YCbCr709.T)


def RGB2020_to_YCbCr2020(RGB):
    m_RGB2020_to_YCbCr2020 = np.array(
        [
            [0.26270000, 0.67800000, 0.05930000],
            [-0.13963006, -0.36036994, 0.50000000],
            [0.50000000, -0.45978570, -0.04021430],
        ]
    )

    return np.matmul(RGB, m_RGB2020_to_YCbCr2020.T)


def writeYUVFile(path, data):
    with open(path, "ab") as f:
        f.write(data.tobytes())
