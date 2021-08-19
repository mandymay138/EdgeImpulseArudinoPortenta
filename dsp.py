import argparse
import numpy as np
import sys
import io, base64
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2


def findfaceinpic(imagein):
    # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    # img = cv2.imread(imagein)
    # gray = cv2.cvtColor(imagein, cv2.COLOR_BGR2GRAY)
    ih, iw = imagein.shape
    gray = imagein
    # gray = cv2.imread('test.jpg')
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    print('gray type:')
    print(type(gray))
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.03,
        minNeighbors=6,
        minSize=(12, 12)
    )
    #  cv2.imshow('img', img)
    if len(faces) == 0:
        # im = Image.new('RGB', (438, 146), (248, 86, 44))
        # draw = ImageDraw.Draw(im)
        # draw.text((10, 10), 'Hello world!', fill=(255, 255, 255))
        print('no face')
        roi_gray = imagein
        # save the image to a buffer, and base64 encode the buffer
        # with io.BytesIO() as buf:
        #     im.save(buf, format='png', bbox_inches='tight', pad_inches=0)
        #     buf.seek(0)
        #     roi_gray = (base64.b64encode(buf.getvalue()).decode('ascii'))
    else:
        print('find face')
        for (x, y, w, h) in faces:
            print ('x y w h', x, y, w, h)
            x = x - 15
            y = y - 15
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            w = w + 30
            h = h + 30
            # roi_gray = cv2.rectangle(imagein, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.rectangle(imagein, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.rectangle(imagein, (90, 90), (80, 80), (0, 0, 0), 3)
            # cv2.rectangle(imagein, (x, y), (x + w, y + h), (0, 0, 0), 3)

            # roi_gray = gray[y:y + h, x:x + w]
            roi_gray = imagein[y:y + h, x:x + w]
            # roi_gray = imagein
            dim = [iw, ih]
            roi_gray = cv2.resize(roi_gray, dim, interpolation=cv2.INTER_AREA)
            roi_color = imagein[y:y + h, x:x + w]
    status = cv2.imwrite('roi1.jpg', roi_gray)
    cv2.destroyAllWindows()
    return roi_gray


def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, channels, extract):
    if (implementation_version != 1):
        raise Exception('implementation_version should be 1')

    # print(type(raw_data))
    print('First raw : shape', raw_data.shape)
    print(raw_data)
    # convert to open cv image before pass to findface
    # open_cv_image = np.array(im)

    # img = np.array(raw_data).reshape(width, height)
    # open_cv_image = np.asarray(img)
    # print('opencvimage :')
    # print(open_cv_image)
    # # open_cv_image = cv2.cvtColor(np.array(pixels), cv2.COLOR_BGR2GRAY)
    # # Convert RGB to BGR
    # # open_cv_image = open_cv_image[:, :, ::-1].copy()
    # image = findfaceinpic(open_cv_image)
    # # convert back to pil image
    # image = Image.fromarray(image)

    graphs = []
    pixels = []
    width = raw_data[0]
    height = raw_data[1]
    raw_data = raw_data[2:].astype(dtype=np.uint32).view(dtype=np.uint8)
    print('2 raw : and shape', raw_data.shape)
    print(raw_data)
    print('raw size 0:')
    print(raw_data.shape[0])
    bs = raw_data.tobytes()
    ix = 0

    # print(type(raw_data))
    while ix < raw_data.shape[0]:
        # ITU-R 601-2 luma transform
        # see: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
        pixels.append(
            (0.299 / 255.0) * float(bs[ix + 2]) + (0.587 / 255.0) * float(bs[ix + 1]) + (0.114 / 255.0) * float(bs[ix]))
        ix = ix + 4

    # PIL image
    # img = Image.fromarray(np.uint8((np.array(pixels) * 255.0).reshape(height, width)), mode='L')
    img = (np.uint8((np.array(pixels) * 255.0).reshape(height, width)))
    print('pil img array from pixels:')
    print(img)
    # open cv image
    # img = np.asarray(img)
    # print('opencv img :')
    # print(img)
    img = findfaceinpic(img)
    raw_data = img.flatten()
    height, width = img.shape
    print(' img height, width:', height, width)
    print(img)
    print(' img flatten  from findface:')
    print(raw_data)
    print('img from findface and raw shape0:', raw_data.shape[0])
    print(img)

    bs = raw_data.tobytes()
    pixels = []
    ix = 0
    while ix < len(bs):
        pixels.append(bs[ix])
        ix = ix + 1
    # if channels == 'Grayscale':
    #     while ix < raw_data.shape[0]:
    #         # ITU-R 601-2 luma transform
    #         # see: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    #         pixels.append((0.299 / 255.0) * float(bs[ix + 2]) + (0.587 / 255.0) * float(bs[ix + 1]) + (0.114 / 255.0) * float(bs[ix]))
    #         ix = ix + 4
    # else:
    #     while ix < raw_data.shape[0]:
    #         pixels.append(float(bs[ix + 2]) / 255.0)
    #         pixels.append(float(bs[ix + 1]) / 255.0)
    #         pixels.append(float(bs[ix]) / 255.0)
    #         ix = ix + 4

    print('pixel size :')
    print(len(pixels))

    im = None
    if channels == 'Grayscale':
        im = Image.fromarray(np.uint8((np.array(pixels) * 255.0).reshape(height, width)), mode='L')
    else:
        im = Image.fromarray(np.uint8((np.array(pixels) * 255.0).reshape(height, width, 3)), mode='RGB')
    # im = Image.fromarray(np.uint8((img * 255.0).reshape(height, width)), mode='L')
    im = im.convert(mode='RGBA')
    buf = io.BytesIO()
    im.save(buf, format='PNG')

    buf.seek(0)
    image = (base64.b64encode(buf.getvalue()).decode('ascii'))
    buf.close()
    if draw_graphs:
        graphs.append({
            'name': 'Image',
            'image': image,
            'imageMimeType': 'image/png',
            'type': 'image'
        })
    # image = np.array(image)
    # image = image.flatten()
    # pixels = []
    # for f in image:
    #     pixels.append(f)
    num_channels = 1
    if channels == 'RGB':
        num_channels = 3

    image_config = {'width': int(width), 'height': int(height), 'channels': num_channels}
    output_config = {'type': 'image', 'shape': image_config}

    # return { 'features': pixels, 'graphs': graphs, 'output_config': output_config }
    return {
        'features': pixels,
        'graphs': graphs,
        'output_config': output_config
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Returns raw data')
    parser.add_argument('--features', type=str, required=True,
                        help='Axis data as a flattened WAV file (pass as comma separated values)')
    parser.add_argument('--axes', type=str, required=True,
                        help='Names of the axis (pass as comma separated values)')
    parser.add_argument('--frequency', type=float, required=True,
                        help='Frequency in hz')
    parser.add_argument('--channels', type=str, required=True,
                        help='Image channels to use.')
    parser.add_argument('--draw-graphs', type=bool, required=True,
                        help='Whether to draw graphs')

    args = parser.parse_args()

    raw_features = np.array([float(item.strip()) for item in args.features.split(',')])
    raw_axes = args.axes.split(',')

    try:
        processed = generate_features(1, False, raw_features, args.axes, args.frequency, args.channels)

        print('Begin output')
        # print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
