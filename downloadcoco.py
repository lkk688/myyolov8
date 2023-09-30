from ultralytics.utils.downloads import download
from pathlib import Path
#import yaml

# Download labels
path=r"E:\Dataset"
segments = True  # segment or box labels
dir = Path(path)  # dataset root dir
url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
download(urls, dir=dir.parent)
# Download data
urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
        'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
        'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
download(urls, dir=dir / 'images', threads=3)
#Downloading https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip to 'E:\coco2017labels-segments.zip'
#Downloading http://images.cocodataset.org/zips/train2017.zip to 'E:\Dataset\images\train2017.zip'
#E:\Dataset\images\test2017\