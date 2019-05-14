import cv2
import numpy as np

def view_dataset(dset):
    cv2.namedWindow('img')
    for idx in range(len(dset)):
        img, bboxes, labels = dset.__getitem__(idx)
        img = img.numpy().transpose(1,2,0)
        bboxes = bboxes.numpy()
        labels = labels.numpy()
        for bbox in bboxes:
            y1, x1, y2, x2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, lineType=1)
        cv2.imshow('img', np.uint8(img))
        k = cv2.waitKey(0)
        if k & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()
