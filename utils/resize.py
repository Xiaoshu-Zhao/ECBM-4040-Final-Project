from os import listdir,makedirs
from os.path import join, isfile, isdir
from PIL import Image


class ImageResizer:
    
    def __init__(self, source_dir, dest_dir):
        
        if not isdir(source_dir):
            raise Exception(f'Input folder does not exists: {source_dir}')
        
        makedirs(dest_dir, exist_ok=True)
        
        self.source_dir = source_dir
        self.dest_dir = dest_dir
                     
    def resize_image(self, filename, size=(299,299)):
        path = join(self.source_dir, filename)
        img = Image.open(path)
        img = img.resize(size)
        img.save(join(self.dest_dir, filename), 'JPEG', optimize=True)
    
    def resize_all(self, size=(299,299)):
        if listdir(self.source_dir)==listdir(self.dest_dir):
            print('Resized images already exist')
            return
        for filename in listdir(self.source_dir):
            img_path = join(self.source_dir, filename)
            if filename.endswith(('.jpg', '.jpeg')) and isfile(img_path):
                self.resize_image(filename, size)
        print(f'Successfully resized all images to {size}, stored at {self.dest_dir}')