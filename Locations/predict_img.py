import os
from PIL import Image
from yolo_cut import YOLO



def get_location(input_img_path,save_img_path):

    print("开始骨折检测")
    yolo = YOLO()
    save_file_path = save_img_path
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    img_detect_path =input_img_path

    img  = Image.open(img_detect_path)
    _,_,outbbox,image_shape = yolo.detect_image(image=img)
    if outbbox is not None:
        #获得h,w,bbox

        left      = int(outbbox[1])     #xmin
        up        = int(outbbox[0])     #ymin
        right     = int(outbbox[3])     #xmax
        below     = int(outbbox[2])     #ymax
        #中心点
        x = int((right-left)/2 + left)
        y = int((below-up)/2   + up)
        #最短边
        lower_line = min(int((right-left)),int((below-up)))
        long_line = max(int((right-left)),int((below-up)))

        need_line = int((long_line+lower_line)/2)
        xmin =int( x - need_line/2)
        ymin = int(y - need_line/2)
        xmax = int(x + need_line/2)
        ymax = int(y + need_line/2)
        #裁剪bbox
        # xmin =int( x - lower_line/2)
        # ymin = int(y - lower_line/2)
        # xmax = int(x + lower_line/2)
        # ymax = int(y + lower_line/2)

        img_cut   =  img.copy()
        crop_img  = img_cut.crop([xmin,ymin,xmax,ymax])
        # crop_img = img_cut.crop([left,up,right,below])
        save_img  = crop_img.resize((256,256))
        save_img.save(save_file_path + "/" + "000000.bmp")


if __name__ == "__main__":
    img_path = r'/Locations/save_out/000000.bmp'
    save_path =r'E:\WZL\AI_Medicine\Fracture_Detections\Locations\save_out'
    get_location(img_path,save_path)