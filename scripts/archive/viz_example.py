import nibabel as nib
import matplotlib.pyplot as plt

example_file = '/path/to/data/inhouse_abdominal_ct/nifti_resampled/CASE0000000/2__ST.nii.gz'

image_data = nib.load(example_file).get_fdata()

center_slice = image_data.shape[2] // 2

image_slice = image_data[:,:,center_slice]

plt.imshow(image_slice, cmap='gray')
plt.savefig('image_example.png')
plt.close()
