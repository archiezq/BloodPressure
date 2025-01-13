import numpy as np

'''
    Lex M. van Loon (lexmaxim.vanloon@anu.edu.au)
    College of Health and Medicine
    Australian National University (ANU)
    Copyright 2020
    
    Schematic overview of the 21-comaprtment model with:
    ---
    |x| -> the compartments, and -x- -> the flows
    ---
                       ---         ---
    ----4--------------|3|----3----|2|--------------2-----
    |                  ---         ---                   |
   ---             #0 Ascending Aorta                   ---
   |4|             #1 Upper thoracic artery             |1|
   ---             #2 Upper body arteries               ---
    |              #3 Upper body veins                   |
    5              #4 Super vena cava                    1
    |              #5 Thoracic aorta                     |
    | ----    ----    ----    ----    ----    ----   --- |
    |-|15|-19-|16|-20-|17|-21-|18|-22-|19|-23-|20|-0-|0|-|
    | ----    ----    ----    ----    ----    ----   --- |
    |              #6 Abdominal aorta                    |
   18              #7 Renal arteries                     6
    |              #8 Renal veins                        |
   ---             #9 Splanchnic arteries               ---
   |4|             #10 Splanchnic veins                 |5|
   ---             #11 Lower body arteries              ---
    |              #12 Lower body veins                  |
   17              #13 Abdominal veins                   7
    |              #14 Inferioir vena cava               |
   ---             #15 Right atrium                     ---
   |4|             #16 Right ventricle                  |6|
   ---             #17 Pulmonoary arteries              ---
    |              #18 Pulmonary veins                   |
    |              #19 Left atrium                       |
    |              #20 left ventricle                    |
    |                   ---         ---                  |
    ----10--------------|8|----9----|7|-------------8-----
    |                   ---         ---                  |
    |                   ---         ---                  |
    ----13--------------|10|---12---|9|------------11-----
    |                   ---         ---                  |
    |                   ---         ---                  |
    ----16--------------|12|---15---|11|-----------14-----
                        ---         ---

    This file holds the parameters for adults resistances, unstressed volumes,
    and compliance for each of the 21 compartments.
    -----------------------------------------------------------------
    
    version 0.0 - initial
    
    -----------------------------------------------------------------
    MIT License
    
    Copyright (c) 2020 Lex M. van Loon
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

class _init_pars(object):
    def __init__(self, scaling):
        # Create empty arrays for all the parameters
        elastance = np.zeros((2,22));
        resistance = np.zeros((6,22)); 
        uvolume = np.zeros((1,22));
        vessel_length = np.zeros((1,22));
    
        """
        Assign the resistances [ mmHg s ml^-1 ]
        
        Note, row 1, 2 and 3 are inflow and row 4, 5 and 6 are outflow resistance
        """
        resistance[:, 0]=[0.007, np.nan, np.nan, 0.003, 0.011, np.nan]       # Ascending Aorta - ADJUSTED 150621 5,0 from 0.03 to 0.011
        resistance[:, 1]=[0.003, np.nan, np.nan, 0.014, np.nan, np.nan]        # Upper thoracic artery
        resistance[:, 1]=[0.003, np.nan, np.nan, 0.014, 0.008, np.nan]        # Upper thoracic artery
        
        resistance[:, 2]=[0.014, np.nan, np.nan, 4.9 * scaling["Rp"] * scaling["Rp_upper"], np.nan, np.nan]         # Upper body arteries
        resistance[:, 3]=[4.9 * scaling["Rp"] * scaling["Rp_upper"], np.nan, np.nan, 0.11, np.nan, np.nan]          # Upper body veins
        
        resistance[:, 4]=[0.11, np.nan, np.nan, 0.028, np.nan, np.nan]         # Super vena cava - ADJUSTED 150621 3, 4 from 0.06 to 0.028
        resistance[:, 5]=[0.011, np.nan, np.nan, 0.01, np.nan, np.nan]        # Lower thoracic artery  - ADJUSTED 150621 0, 5 from 0.03 to 0.011
        resistance[:, 6]=[0.01, np.nan, np.nan, 0.1, 0.03, 0.09]            # Abdominal aorta - ADJUSTED 150621 0, 5 from 0.03 to 0.01
        resistance[:, 7]=[0.1, np.nan, np.nan, 4.1*scaling["Rp"], np.nan, np.nan]         # Renal arteries
        resistance[:, 8]=[4.1*scaling["Rp"], np.nan, np.nan, 0.11, np.nan, np.nan]         # Renal veins
        resistance[:, 9]=[0.03, np.nan, np.nan, 3*scaling["Rp"], np.nan, np.nan]          # Splanchnic arteries
        resistance[:, 10]=[3*scaling["Rp"], np.nan, np.nan, 0.07, np.nan, np.nan]          # Splanchnic veins
        resistance[:, 11]=[0.09, np.nan, np.nan, 4.5*scaling["Rp"], np.nan, np.nan]         # Lower body arteries
        resistance[:, 12]=[4.5*scaling["Rp"], np.nan, np.nan, 0.1, np.nan, np.nan]          # Lower body veins
        resistance[:, 13]=[0.11, 0.07, 0.1, 0.019, np.nan, np.nan]             # Abdominal veins
        resistance[:, 14]=[0.019, np.nan, np.nan, 0.008, np.nan, np.nan]     # Inferior vena cava
        resistance[:, 15]=[0.008, 0.028, np.nan, 0.006, np.nan, np.nan]       # Right atrium
        resistance[:, 16]=[0.006, np.nan, np.nan, 0.003, np.nan, np.nan]     # Right ventricle
        resistance[:, 17]=[0.003, np.nan, np.nan, 0.07*scaling["Rp"]*scaling["Rp_lungs"], np.nan, np.nan]      # Pulmonary artery
        resistance[:, 18]=[0.07*scaling["Rp"]*scaling["Rp_lungs"], np.nan, np.nan, 0.006, np.nan, np.nan]       # Pulmonary veins
        resistance[:, 19]=[0.006, np.nan, np.nan, 0.01, np.nan, np.nan]     # Left atrium
        resistance[:, 20]=[0.01, np.nan, np.nan, 0.007, np.nan, np.nan]    # Left ventricle
        resistance[:, 21]=[0.008,np.nan,np.nan,0.016,np.nan,np.nan];    # Carotid artery
        resistance = np.array(resistance)*scaling["Global_R"]

        """
        resistance[:,0]=[0.007,np.nan,np.nan,0.003,0.011,np.nan];         # Ascending Aorta - ADJUSTED 150621 5,0 from 0.03 to 0.011
        #resistance[:,1]=[0.003,np.nan,np.nan,0.014,0.014,np.nan];        # Upper thoracic artery
        #resistance[:,2]=[0.014,np.nan,np.nan,4.9,np.nan,np.nan];         # Upper body arteries
        resistance[:,1]=[0.003,np.nan,np.nan,0.112,0.014,np.nan];        # Upper thoracic artery
        resistance[:,2]=[0.112,np.nan,np.nan,4.9,np.nan,np.nan];         # Upper body arteries
        resistance[:,3]=[4.9,np.nan,np.nan,0.11,np.nan,np.nan];          # Upper body veins
        resistance[:,4]=[0.11,np.nan,np.nan,0.028,np.nan,np.nan];         # Super vena cava - ADJUSTED 150621 3,4 from 0.06 to 0.028
        resistance[:,5]=[0.011,np.nan,np.nan,0.01,np.nan,np.nan];        # Lower thoracic artery  - ADJUSTED 150621 0,5 from 0.03 to 0.011
        resistance[:,6]=[0.01,np.nan,np.nan,0.1,0.03,0.09];            # Abdominal aorta - ADJUSTED 150621 0,5 from 0.03 to 0.01
        resistance[:,7]=[0.1,np.nan,np.nan,4.1,np.nan,np.nan];         # Renal arteries
        resistance[:,8]=[4.1,np.nan,np.nan,0.11,np.nan,np.nan];         # Renal veins
        resistance[:,9]=[0.03,np.nan,np.nan,3,np.nan,np.nan];          # Splanchnic arteries
        resistance[:,10]=[3,np.nan,np.nan,0.07,np.nan,np.nan];          # Splanchnic veins
        resistance[:,11]=[0.09,np.nan,np.nan,4.5,np.nan,np.nan];         # Lower body arteries
        resistance[:,12]=[4.5,np.nan,np.nan,0.1,np.nan,np.nan];          # Lower body veins
        resistance[:,13]=[0.11,0.07,0.1,0.019,np.nan,np.nan];             # Abdominal veins
        resistance[:,14]=[0.019,np.nan,np.nan,0.008,np.nan,np.nan];       # Inferior vena cava
        
        resistance[:,15]=[0.008,0.028,np.nan,0.006,np.nan,np.nan];       # Right atrium
        resistance[:,16]=[0.006,np.nan,np.nan,0.003,np.nan,np.nan];     # Right ventricle
        resistance[:,17]=[0.003,np.nan,np.nan,0.07,np.nan,np.nan];      # Pulmonary artery
        resistance[:,18]=[0.07,np.nan,np.nan,0.006,np.nan,np.nan];       # Pulmonary veins
        resistance[:,19]=[0.006,np.nan,np.nan,0.01,np.nan,np.nan];     # Left atrium
        resistance[:,20]=[0.01,np.nan,np.nan,0.007,np.nan,np.nan];    # Left ventricle
        """
        """
        Assign the elastances [ mmHg ml^-1 ]
        
        Note, row 1 is maximum elastance, and row 2 is minimum elastance (i.e. systolic and diastolic).
        """
        elastance[:,0]=[1/.28,np.nan];           # Ascending Aorta
        elastance[:,1]=[1/.13,np.nan];           # Upper thoracic artery
        
        elastance[:,2]=[1/.2,np.nan];           # Upper body arteries- ADJUSTED 150621 from .42 to .2
        elastance[:,3]=[1/7,np.nan];            # Upper body veins - ADJUSTED 150621 from 11 to 7
        
        elastance[:,4]=[1/1.3,np.nan];            # Super vena cava
        elastance[:,5]=[1/.1,np.nan];           # Lower thoracic artery
        elastance[:,6]=[1/.1,np.nan];           # Abodiminal aorta
        elastance[:,7]=[1/.21,np.nan];           # Renal arteries
        elastance[:,8]=[1/5,np.nan];            # Renal veins
        elastance[:,9]=[1/.2,np.nan];          # Splanchnic arteries
        elastance[:,10]=[1/65,np.nan];          # Splanchnic veins
        elastance[:,11]=[1/.2,np.nan];          # Lower body arteries
        elastance[:,12]=[1/22,np.nan];          # Lower body veins
        elastance[:,13]=[1/1.3,np.nan];          # Abdominal veins
        elastance[:,14]=[1/.5,np.nan];          # Inferior vena cava
        
        elastance[:,15]=[1/1.35,1/3.33];         # Right atrium
        elastance[:,16]=[1/.77,1/19.29] * np.array([scaling["max_RV_E"], scaling["min_RV_E"]]);        # Right ventricle
        
        elastance[:,17]=[1/3.4,np.nan];         # Pulmonary artery
        elastance[:,18]=[1/9,np.nan];         # Pulmonary veins
        
        elastance[:,19]=[1/1.64,1/2];           # Left atrium
        elastance[:,20]=[1/.4,1/9.69] * np.array([scaling["max_LV_E"], scaling["min_LV_E"]]);          # Left ventricle

        elastance[:,21]=[1/0.2,np.nan]         # Carotid artery

        inputs = np.ones(22)
        inputs[0:3] = inputs[5:8] = inputs[9] = inputs[11] = inputs[17] = scaling["E_arteries"]
        inputs[3:5] = inputs[8] = inputs[10] = inputs[12:15] = inputs[18] = scaling["E_veins"]
        elastance = np.array(elastance)*inputs # heart elastanceses are included,
        

        """
        Assign the unstressed volumes [ ml ]
        
        """
        uvolume[:,0]=[21];           # Ascending Aorta
        uvolume[:,1]=[5];           # Upper thoracic artery
        
        uvolume[:,2]=[16];           # Upper body arteries - ADJUSTED 150621 from 200 to 16
        uvolume[:,3]=[645];            # Upper body veins
        
        uvolume[:,4]=[16];            # Super vena cava
        uvolume[:,5]=[200];           # Lower thoracic artery - ADJUSTED 150621 from 16 to 200
        uvolume[:,6]=[10];           # Abodiminal aorta
        uvolume[:,7]=[20];           # Renal arteries
        uvolume[:,8]=[30] * scaling["venous_UV"];            # Renal veins
        uvolume[:,9]=[300];          # Splanchnic arteries
        uvolume[:,10]=[1146] * scaling["venous_UV"];          # Splanchnic veins
        uvolume[:,11]=[200];          # Lower body arteries
        uvolume[:,12]=[716] * scaling["venous_UV"];          # Lower body veins
        uvolume[:,13]=[79];          # Abdominal veins
        uvolume[:,14]=[33];          # Inferior vena cava
        
        uvolume[:,15]=[14];        # Right atrium
        uvolume[:,16]=[36];        # Right ventricle
        uvolume[:,17]=[160];         # Pulmonary artery
        uvolume[:,18]=[430];         # Pulmonary veins
        uvolume[:,19]=[11];           # Left atrium
        uvolume[:,20]=[20];          # Left ventricle
        
        uvolume[:,21]=[10];         # Carotid artery, source?

        uvolume = np.array(uvolume) * scaling["Global_UV"]

        """
        Assign the vessel lengths [cm], (NOT relative to the heart, CHECK THIS)
        """
        vessel_length[:,0]=[10];           # Ascending Aorta
        vessel_length[:,1]=[4.5];           # Upper thoracic artery
        
        vessel_length[:,2]=[20];           # Upper body arteries
        vessel_length[:,3]=[20];            # Upper body veins
        
        vessel_length[:,4]=[4.5];            # Super vena cava
        vessel_length[:,5]=[16];           # Lower thoracic artery
        vessel_length[:,6]=[14.5];           # Abodiminal aorta
        vessel_length[:,7]=[0];           # Renal arteries
        vessel_length[:,8]=[0];            # Renal veins
        vessel_length[:,9]=[5];          # Splanchnic arteries
        vessel_length[:,10]=[5];          # Splanchnic veins
        vessel_length[:,11]=[106];          # Lower body arteries
        vessel_length[:,12]=[106];          # Lower body veins
        vessel_length[:,13]=[14.5];          # Abdominal veins
        vessel_length[:,14]=[6];          # Inferior vena cava
        
        vessel_length[:,15]=[0];          # Right atrium
        vessel_length[:,16]=[0];          # Right ventricle
        
        vessel_length[:,17]=[0];          # Pulmonary arteries, for now zero because the appropiate model is not completed
        vessel_length[:,18]=[0];          # Pulmonary veins, for now zero because the appropiate model is not completed
        
        vessel_length[:,19]=[0];          # Left atrium
        vessel_length[:,20]=[0];          # Left ventricle

        vessel_length[:,21]=[21.5];          # Carotid artery
           

        
        self.resistance = resistance;
        self.elastance = elastance;
        self.uvolume = uvolume;
        self.vessel_length = vessel_length;
        




