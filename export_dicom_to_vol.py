"""
Created: 2018-07-27
@author: Mirabela Rusu
example how to run:
python  export_dicom_to_vol.py --in_path C:\path_to_dicom_folder --out_fn C:\out.nii.gz
""" 

from __future__ import print_function
import argparse
import SimpleITK as sitk
import os
import numpy as np
import cv2
import scipy
import scipy.misc

def get_vol_from_dicom_dir(dicom_dir, verbose=False):
    """
    goes through all dicom files in dicom_dir, finds the series (as it could 
    be multiple), and get the volume for each series
    """

    if verbose:
        print("Processing folder: ", dicom_dir)
    
    reader = sitk.ImageSeriesReader()    
    series_found = reader.GetGDCMSeriesIDs(dicom_dir)

    series_found_count = 0
    sitk_im_list = []
    for serie in series_found:
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, serie)
        if len(dicom_names):
            print("  Found series", serie)
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_names)
            sitk_im = reader.Execute()
            sitk_im_list.append(sitk_im)
            
    if verbose:
        print ("Found", len(sitk_im_list), "series")

    return sitk_im_list
    
def write_im_list(im_list, out_fn, verbose):
    # for than one series was found: each file needs a new name
    if len(im_list)>1: 
        print("Writing multiple series")
        filename, file_extension = os.path.splitext(out_fn)
        #in case one uses nii.gz
        if file_extension in ['.gz', '.bz2']:
            filename_new, file_extension_new = os.path.splitext(filename)
            file_extension = file_extension_new + file_extension
            filename = filename_new
        
        out_fn = ["{:s}_{:02d}{:s}".format(filename, i, file_extension) 
            for i in range(len(im_list))]
    else:
        out_fn = [out_fn]
        filename, file_extension = os.path.splitext(out_fn)
        
    
    for im, fn in zip(im_list, out_fn):
        if verbose:
            print("  Writing:", fn)
        sitk.WriteImage(im, fn)
        
    return filename, file_extension
    
def example_export_one_plane(im, file_name,verbose=False):#, file_101, file_110, verbose=False):
    
    # show sitk image info
    print("SITK: ImageSize:     ", im.GetSize())    
    print("SITK: ImageVoxelSize ", im.GetSpacing())    
    
    #get the python ndarray
    #im_shape=im.GetSize() 
    #print('im_shape: ',im_shape)
    #new_shape=[im_shape[2]*8,im_shape[2]*8,im_shape[2]]
    #print('new shape: ', new_shape)
    
    #ref_im=im
    #im = sitk.Resample(im,ref_im,sitk.Transform())
    #print('new im_shape: ',im.GetSize())

    arr = sitk.GetArrayFromImage(im)
    sp = [im.GetSpacing()[2],im.GetSpacing()[0], im.GetSpacing()[1]]
    print(arr.shape)
    #exit()
    #! notice reorder of axes (shape) compare to sitk_im 
    print("NdArray: ImageSize:  ", arr.shape, "Voxel Size ", sp)  
    
    
    print("NdArray: ImageSize(0): ", arr[0,:,:].shape, "Pixel Size ", sp[1:3])
    print("NdArray: ImageSize(1): ", arr[:,0,:].shape, "Pixel Size ", 
        [sp[0],sp[2]])
    print("NdArray: ImageSize(2): ", arr[:,:,0].shape, "Pixel Size ", sp[0:2])
    
    """ example how to export image (png).
    NOTE: It needs rescaling between 0-255 
    """
    
     #scale between 0-255
    im = np.interp(arr, (arr.min(), arr.max()), (0, 255)).astype('uint8')
    print("Range Intensity Before: ", arr.min(), arr.max())
    print("Range Intensity After:  ", im.min(), im.max())
    print('after interp:', im.shape)
    
    #reshape to get 8x difference
   # im = np.resize(im, [arr.shape[0],arr.shape[0]*8,arr.shape[0]*8]).astype('uint8')
    #print('after resize:', im.shape)    
    
    mid_slice_x = int(arr.shape[1]/2)
    mid_slice_y = int(arr.shape[2]/2)
    mid_slice_z = int(arr.shape[0]/2)
    print("Range Intensity After(Sl)", im[:,mid_slice_x,:].min(), im[:,mid_slice_x,:].max())
    
    #if verbose:
        #print('Writing file', fn+".png")
    #below used to be fn+'.png'
   # new_shape=[arr.shape[0],arr.shape[0]*8,arr.shape[0]*8]
   # new_im=np.resize(im,new_shape).astype('uint8')
   # cv2.imwrite("ax_cor.png",new_im[:,mid_slice_x,:])
   # cv2.imwrite('ax_sag.png', im[:,:,mid_slice_y])
   # cv2.imwrite('ax.png',im[mid_slice_z,:,:])
 
    for i in range(arr.shape[0]):
        cv2.imwrite(file_name+'/'+str(i)+'.png',im[i,:,:])

    #ax_cor UNCOMMENT THIS 
    #for i in range(arr.shape[1]):
     #   cv2.imwrite(file_101+'/'+str(i)+'.png',im[:,i,:]) 
    #ax_sag
    #for i in range(arr.shape[2]):
     #   cv2.imwrite(file_110+'/'+str(i)+'.png',im[:,:,i])
    #cor/sag depending on current directory    
    #for i in range(arr.shape[0]):
    #    cv2.imwrite('./cor/'+str(i)+'.png',im[i,:,:])
    #test image
    #cv2.imwrite('test.png',im[:,20,:])


def main():
    """Main
    """
    # Parse data settings
    parser = argparse.ArgumentParser(description='Parse data')
    parser.add_argument('--verbose', action='store_false',
                        help='verbose output')
    parser.add_argument('--in_path', type=str, required=True, default=".",
                        help="One folder contain dicom files")
    parser.add_argument('--out_fn', type=str, required=False, default =
        "out.nii.gz", help='output filename. WARNING: It will be rewrote if it exists')
    #UNCOMMENT THESE
    #parser.add_argument('--file_101', type=str, required=True, default='./file_101', help='location for im[:,i,:] pngs')
    #parser.add_argument('--file_110', type=str, required=True, default='./file_110', help='location for im[:,:,i] pngs')
    opt = parser.parse_args()
    if opt.verbose:
        print(opt)
    
    #it could happen that multiple series exist in one dicom folder:
    #parse and return a volume for each series
    sitk_im_list = get_vol_from_dicom_dir(opt.in_path, opt.verbose)
    
    #write the volumes
    #fn, ext = write_im_list(sitk_im_list, opt.out_fn, opt.verbose)
    fn='img'
    #code only used as example to show how to access planes and other info, #only show for the first series
    example_export_one_plane(sitk_im_list[0], opt.out_fn, opt.verbose)#opt.file_101, opt.file_110, opt.verbose)
    
   

if __name__ == "__main__":
    main()
    
