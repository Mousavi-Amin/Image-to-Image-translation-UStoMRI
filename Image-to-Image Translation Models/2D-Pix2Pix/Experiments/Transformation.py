import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from Tools import metrics
from batchup import data_source
import nibabel as nib
from nibabel.processing import conform
from Models.Pix2Pix2D import Pix2Pix
import utils
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import transform

class Experiment_transformation():
    def __init__(self, preprocessed_dataset_path, network_structure='Pix2Pix', base_project_dir='.', args={}, mode='train'):
        self.preprocessed_dataset_path = preprocessed_dataset_path
        self.network_structure = network_structure
        self.base_project_dir = base_project_dir
        self.args = args
        self.task = ''
        self.mode = mode
        
        plt.switch_backend('agg')
        
        #assert tf.test.is_gpu_available()
        #assert tf.test.is_built_with_cuda()
        
        print('[INFO] GPU is enabled.')

        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def prepare(self):

        self.train_dataset, self.val_dataset, self.test_dataset, self.file_names = utils.create_dataset(self.preprocessed_dataset_path, self.args['val_ratio'], self.args['test_ratio'], True, self.args['random_seed'])

        control_slice_dir = os.path.join(self.base_project_dir, 'ControlSlice')
        control_slice_name = os.listdir(control_slice_dir)[0]
        control_slice_path = os.path.join(control_slice_dir, control_slice_name)
        data = np.load(control_slice_path, allow_pickle=True) 
        self.control_US, self.control_MRI =  data['us'], data['mri']

        self.args['input_shape'] = list(self.control_US.shape[:2] ) + [1]
        print('Input Shape:', self.args['input_shape'])
        print('\nTrain size:', len(self.train_dataset))
        print('Val size:', len(self.val_dataset))
        print('Test size:', len(self.test_dataset))
        
        self.task += self.network_structure 
        
        output_dir = os.path.join(self.base_project_dir, 'Output', self.task)
        output_models_dir = os.path.join(output_dir, 'TrainedModels')
        self.args['best_models_dir'] = os.path.join(output_models_dir, 'BestModels')
        self.args['last_models_dir']= os.path.join(output_models_dir, 'LastModels')
        self.args['log_dir'] = os.path.join(output_dir, 'Log')
        self.args['sample_generated_images_dir'] = os.path.join(output_dir, 'SampleGeneratedImages')

        if self.network_structure == 'Pix2Pix':
            self.model = Pix2Pix(args=self.args)


        if os.path.exists(os.path.join(self.args['last_models_dir'], 'Generator.h5')):
            print('[INFO] Loading pretrained model weights from: ' + self.args['last_models_dir'])
            pretrained_generator_path = os.path.join(self.args['last_models_dir'], 'Generator.h5')
            pretrained_discriminator = os.path.join(self.args['last_models_dir'], 'Discriminator.h5')
            self.model.generator.load_weights(pretrained_generator_path)
            self.model.discriminator.load_weights(pretrained_discriminator)
        else:
            print('[INFO] Pretrained weights not found. training will be strated using randomly intialized weights.')

        print('Generator:')
        self.model.generator.summary()
        print('\n\nDiscriminator:')
        self.model.discriminator.summary()
        
        Path(self.args['best_models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.args['last_models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.args['sample_generated_images_dir']).mkdir(parents=True, exist_ok=True)

        print("Task: " + self.task)

        
    def train(self):

        print("[INFO] Pix2Pix training initiated ...")
        
        try:
            self.model.fit(
                train_data = self.train_dataset,
                control_US = self.control_US,
                control_MRI=self.control_MRI
            )
        except KeyboardInterrupt:
            pass
        

    def generate_output_2D(self):
        output_path = os.path.join(self.base_project_dir, 'Output', self.task, 'Results', '2D')
        Path(output_path).mkdir(parents=True, exist_ok=True)
        for set_name in ['train', 'val', 'test']:
            for result_dir_name in ['GeneratedImages', 'DifferentialImages', 'ResultedMetrics']:
                result_dir_path = os.path.join(output_path, set_name, result_dir_name)
                Path(result_dir_path).mkdir(parents=True, exist_ok=True)

        print('Loading model weights from: ' + self.args['last_models_dir'])
        self.model.generator.load_weights(os.path.join(self.args['last_models_dir'], 'Generator.h5'))

        print('Generating outputs ...')
        for set_name in ['train', 'val', 'test']:
            generated_images_path = os.path.join(output_path, set_name, 'GeneratedImages')
            differential_images_path = os.path.join(output_path, set_name, 'DifferentialImages')
            resulted_metrics_path = os.path.join(output_path, set_name, 'ResultedMetrics')
            
            df_columns_names = ['Patient_ID', 'MAE', 'MAPE', 'RMSE', 'SSI']#, 'PSNR']
            df_metrics = pd.DataFrame(columns=df_columns_names)

            if set_name == 'train':
                dataset = self.train_dataset
            elif set_name == 'val':
                dataset = self.val_dataset
            elif set_name == 'test':
                dataset = self.test_dataset

            for i, (US, gt_MRI) in tqdm(enumerate(dataset)):
                file_name = self.file_names[set_name][i].split(os.sep)[-1].replace('.npz', '')
                US = US.numpy()
                gt_MRI = gt_MRI.numpy()

                pred_MRI = self.model.generator(np.expand_dims(US, axis=0), training=True).numpy()
                pred_MRI = np.squeeze(pred_MRI, axis=0)

                pred_MRI_path = os.path.join(generated_images_path, file_name+ '.png')
                plt.imshow(pred_MRI)
                plt.savefig(pred_MRI_path)
                plt.close()

                diff_MRI = gt_MRI - pred_MRI
                diff_MRI_path = os.path.join(differential_images_path, file_name + '.png')
                plt.imshow(diff_MRI)
                plt.savefig(diff_MRI_path)
                plt.close()

                mae = metrics.Mean_absolute_error(gt_MRI, pred_MRI)
                mape = metrics.Mean_absolute_percentage_error(gt_MRI, pred_MRI)
                rmse = metrics.Root_mean_squared_error(gt_MRI, pred_MRI)
                ssi = metrics.Structural_similarity(gt_MRI, pred_MRI)
                #psnr = metrics.Peak_signal_noise_ratio(gt_MRI, pred_MRI)
                #psnr = 0
                row = pd.Series([file_name, mae, mape, rmse, ssi], index=df_metrics.columns)
                df_metrics = df_metrics.append(row, ignore_index=True)
            
           
            df_metrics_path = os.path.join(resulted_metrics_path, 'CalculatedMetrics.csv')
            df_metrics.to_csv(df_metrics_path)


    def generate_output_3D(self):
        output_path = os.path.join(self.base_project_dir, 'Output', self.task, 'Results', '3D')
        Path(output_path).mkdir(parents=True, exist_ok=True)
        for set_name in ['train', 'val', 'test']:
            for result_dir_name in ['GeneratedImages', 'DifferentialImages', 'ResultedMetrics']:
                result_dir_path = os.path.join(output_path, set_name, result_dir_name)
                Path(result_dir_path).mkdir(parents=True, exist_ok=True)

        print('[INFO] Loading model weights from: ' + self.args['last_models_dir'])
        self.model.generator.load_weights(os.path.join(self.args['last_models_dir'], 'Generator.h5'))

        #images_metadata = utils.get_volume_metadata(os.path.join(self.preprocessed_dataset_path, '..', 'MRI_Images'))

        print('[INFO] Generating 3D outputs based on 2D predictions ...')
        for set_name in ['train', 'val', 'test']:
            generated_images_path = os.path.join(output_path, set_name, 'GeneratedImages')
            differential_images_path = os.path.join(output_path, set_name, 'DifferentialImages')
            resulted_metrics_path = os.path.join(output_path, set_name, 'ResultedMetrics')
            
            df_columns_names = ['Patient_ID', 'MAE', 'MAPE', 'RMSE', 'SSI']#, 'PSNR']
            df_metrics = pd.DataFrame(columns=df_columns_names)

            if set_name == 'train':
                dataset = self.train_dataset
            elif set_name == 'val':
                dataset = self.val_dataset
            elif set_name == 'test':
                dataset = self.test_dataset

            # find volumes names
            set_vol_names = []
            for  i in range(len(dataset)):
                file_name = self.file_names[set_name][i].split(os.sep)[-1][9:15] 
                set_vol_names.append(file_name)
                
            set_vol_names = list(set(set_vol_names))

            for vol_name in tqdm(set_vol_names):
                vol_slices_path = os.path.join(self.preprocessed_dataset_path, vol_name)
                vol_slices_names = os.listdir(vol_slices_path)
                print(vol_slices_path)
                images_metadata = utils.get_volume_metadata(os.path.join('Z:\GAN Data_Prostate cancer\Algorithms\Pix2Pix 2D_Versions\Pix2Pix_2D_Ver5\Data\MRI_Images', vol_name[:4] + '_MRI_Image_{}.nii.gz'.format(vol_name[5])))
                #images_metadata = utils.get_volume_metadata(vol_slices_path)
                gt_MRI_list = []
                pred_MRI_list = []
                for slice_name in vol_slices_names:
                    slice_path = os.path.join(vol_slices_path, slice_name)
                    data = np.load(slice_path, allow_pickle=True) 
                    US, gt_MRI =  data['us'], data['mri']
                    
                    pred_MRI = self.model.generator(np.expand_dims(US, axis=0), training=True).numpy()
                    pred_MRI = np.squeeze(pred_MRI)
                    gt_MRI = np.squeeze(gt_MRI)

                    gt_MRI_list.append(gt_MRI)
                    pred_MRI_list.append(pred_MRI)
                    
                gt_MRI_volume = np.stack(gt_MRI_list, axis=-1)
                pred_MRI_volume = np.stack(pred_MRI_list, axis=-1)

                pred_MRI_volume_path = os.path.join(generated_images_path, vol_name)
                pred_nii_img = nib.Nifti1Image(pred_MRI_volume, images_metadata['affine'], images_metadata['header'])
                #a = nib.aff2axcodes(images_metadata['affine'])
                a = nib.aff2axcodes(images_metadata['affine'])
                delim = ""
                #print (vol_name)
                a = delim.join(map(str, a))
                #print( "   ", images_metadata['header'])
                print(images_metadata['header'].get_data_shape())
                v = (abs(images_metadata['affine'][0, 0]), abs(images_metadata['affine'][1, 1]), abs(images_metadata['affine'][2, 2]))
                pred_nii_img = conform(pred_nii_img, out_shape=images_metadata['header'].get_data_shape(), voxel_size=v, order=3, cval=0.0, orientation=a, out_class=None)
               
                print(pred_MRI_volume_path.split('\\')[-1])
                nib.save(pred_nii_img, os.path.join(pred_MRI_volume_path))
                
                diff_MRI_volume = gt_MRI_volume - pred_MRI_volume
                diff_MRI_volume_path = os.path.join(differential_images_path, vol_name)
                diff_nii_img = nib.Nifti1Image(diff_MRI_volume, images_metadata['affine'], images_metadata['header'])
                diff_nii_img = conform(diff_nii_img, out_shape=(128, 128, 64), voxel_size=(1.0, 1.0, 1.0), order=3, cval=0.0, orientation=a, out_class=None)

                nib.save(diff_nii_img, diff_MRI_volume_path)

                mae = metrics.Mean_absolute_error(gt_MRI, pred_MRI)
                mape = metrics.Mean_absolute_percentage_error(gt_MRI, pred_MRI)
                rmse = metrics.Root_mean_squared_error(gt_MRI, pred_MRI)
                ssi = metrics.Structural_similarity(gt_MRI, pred_MRI)
                # psnr = metrics.Peak_signal_noise_ratio(gt_MRI, pred_MRI)
                # psnr = 0

                row = pd.Series([file_name, mae, mape, rmse, ssi], index=df_metrics.columns)
                df_metrics = df_metrics.append(row, ignore_index=True)
            
            df_metrics_path = os.path.join(resulted_metrics_path, 'CalculatedMetrics.csv')
            df_metrics.to_csv(df_metrics_path)

