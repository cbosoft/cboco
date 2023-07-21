import numpy as np
import cv2

class Image:

    def __init__(
            self,
            id: int,
            file_name: str,
            width: int,
            height: int,
            **extra):
        self.id = id

        # only forward slashes are allowed here.
        self.file_name = file_name.replace('\\', '/')
        self.width = width
        self.height = height
        self.extra = extra
        self.annotations = []
        fn_parts = self.file_name.split('/')
        self.base_name = fn_parts[-1]
        self.hashable_name = '/'.join(fn_parts[-3:])
    
    @classmethod
    def from_file(cls, file_name: str, **extra) -> "Image":
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f'Could not read image "{file_name}".')
        h, w = img.shape
        return cls(
            id=-1,
            file_name=file_name,
            width=w,
            height=h,
            **extra
        )
    
    def set_id(self, v: int):
        self.id = v
        for ann in self.annotations:
            ann.image_id = v
    
    def add_annotation(self, annotation):
        annotation.image_id = self.id
        self.annotations.append(annotation)
        return annotation
    
    def to_dict(self) -> dict:
        return dict(
            id=self.id,
            width=self.width,
            height=self.height,
            file_name=self.file_name,
            **self.extra
        )
    
    def __eq__(self, other: "Image") -> bool:
        if isinstance(other, str):
            return self.base_name == other
        
        i_parts = self.file_name.split('/')[::-1]
        o_parts = other.file_name.split('/')[::-1]
        same = 0
        for i, o in zip(i_parts, o_parts):
            if i != o:
                break
            same += 1
        return same >= 1
    
    def __hash__(self) -> int:
        return hash(self.hashable_name)