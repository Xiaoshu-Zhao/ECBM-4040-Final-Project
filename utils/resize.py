from os import listdir,makedirs
from os.path import join, isfile, isdir
from PIL import Image
import numpy as np


class ImageResizer:
    
    def __init__(self, source_dir, dest_dir):
        
        if not isdir(source_dir):
            raise Exception(f'Input folder does not exists: {source_dir}')
        
        makedirs(dest_dir, exist_ok=True)
        
        self.source_dir = source_dir
        self.dest_dir = dest_dir
    

    def is_RGB(self,img_path):
        """
        check if image is rgb
        only accept rgb image when building train set
        """
        image=Image.open(img_path)
        image=np.asarray(image)
        if(len(image.shape)<3):
            return False
        return True
                     
    def resize_image(self, filename, size=(299,299,3)):
        """
        save resized image to destination path
        """
        path = join(self.source_dir, filename)
        img = Image.open(path)
        img = img.resize(size)
        img.save(join(self.dest_dir, filename), 'JPEG', optimize=True)
    
    def resize_all(self, size=(299,299,3)):
        """
        resize all image
        """
        if listdir(self.dest_dir) != None:
            print('Resized images already exist')
            return
        count=0
        for filename in listdir(self.source_dir):
            img_path = join(self.source_dir, filename)
            if filename.endswith(('.jpg', '.JPEG','jpeg')) and isfile(img_path) and self.is_RGB(img_path):
                self.resize_image(filename, size)
                count+=1
        print(f'Successfully resized {count} images to {size}, stored at {self.dest_dir}')