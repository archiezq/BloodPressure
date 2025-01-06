import numpy as np

'''
Lex M. van Loon (lexmaxim.vanloon@anu.edu.au)
College of Health and Medicine
Australian National University (ANU)

MIT License, Copyright (c) 2020 Lex M. van Loon

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

This file holds the parameters for reflexes
-----------------------------------------------------------------
version 0.0 - initial (14-10-2020)

TODO:
    - Added references to this file 
-----------------------------------------------------------------'''

class _init_reflex(object):
    def __init__(self, scaling):

        
        """
            Create impulse responses
        """
        
        S_GRAN = 0.0625;
        SUMZ = 0;
        I_length=960;
        S_INT = .25;
        p=np.zeros(I_length);
        s=np.zeros(I_length);
        v=np.zeros(I_length);
        a=np.zeros(I_length);
        cpv=np.zeros(I_length);
        cpa=np.zeros(I_length);
        
        # Set up parasympathetic impulse response function
        start = int(np.round((scaling["baro_delay_para"] + .59 - 3.0*S_GRAN) / S_GRAN)); #vec[25] // Delay of the parasymp. impulse response
        peak  = int(np.round((scaling["baro_delay_para"] + .70 - 3.0*S_GRAN) / S_GRAN)); #vec[26] // Peak of the parasymp. impulse response
        end   = int(np.round((scaling["baro_delay_para"] + scaling["baro_stretch_para"] + 1 - 3.0*S_GRAN) / S_GRAN)); #vec[27] // End of the parasymp. impulse response
        SUMZ = 0
        for i in range(start,peak):
            p[i]=(i-start)/(peak-start)
            SUMZ=SUMZ+p[i]
        for i in range(peak,end):
            p[i]=(end-i)/(end-peak)
            SUMZ=SUMZ+p[i]
        for i in range(start,end):
            p[i]=p[i]/(SUMZ)
        
        # Do the same for the heldtbeta-sympathetic impulse response function.
        start = int(np.round((scaling["baro_delay_sympa"] + 2.5 - 3.0*S_GRAN) / S_GRAN)); #vec[28] // Delay of the parasymp. impulse response
        peak  = int(np.round((scaling["baro_delay_sympa"] + 3.5 - 3.0*S_GRAN) / S_GRAN)); #vec[29] // Peak of the parasymp. impulse response
        end   = int(np.round((scaling["baro_delay_sympa"] + scaling["baro_stretch_sympa"] + 15 - 3.0*S_GRAN) / S_GRAN)); #vec[30] // End of the parasymp. impulse response
        SUMZ = 0
        for i in range(start,peak):
            s[i]=(i-start)/(peak-start)
            SUMZ=SUMZ+s[i]
        for i in range(peak,end):
            s[i]=(end-i)/(end-peak)
            SUMZ=SUMZ+s[i]
        for i in range(start,end):
            s[i]=s[i]/(SUMZ)
                    
        # Do the same for the venous alpha-sympathetic impulse response function.
        start = int(np.round((scaling["baro_delay_BR_R"] + 5 - 3.0*S_GRAN) / S_GRAN)); #vec[112] // Delay of the alpha-sympathetic impulse response
        peak  = int(np.round((scaling["baro_delay_BR_R"] + 10 - 3.0*S_GRAN) / S_GRAN)); #vec[113] // Peak of the alpha-sympathetic impulse response
        end   = int(np.round((scaling["baro_delay_BR_R"] + scaling["baro_stretch_BR_R"] + 42 - 3.0*S_GRAN) / S_GRAN)); #vec[114] // End of the alpha-sympathetic impulse response
        SUMZ=0
        for i in range(start,peak):
            v[i]=(i-start)/(peak-start)
            SUMZ=SUMZ+v[i]
        for i in range(peak,end):
            v[i]=(end-i)/(end-peak)
            SUMZ=SUMZ+v[i]
        for i in range(start,end):
            v[i]=v[i]/(SUMZ)
                            
        # Do the same for the arterial alpha-sympathetic impulse response function.
        start = int(np.round((scaling["baro_delay_BR_UV"] + 2.5 - 3.0*S_GRAN) / S_GRAN)); #vec[115] // Delay  impulse response
        peak  = int(np.round((scaling["baro_delay_BR_UV"] + 3.5 - 3.0*S_GRAN) / S_GRAN)); #vec[116] // Peak  impulse response
        end   = int(np.round((scaling["baro_delay_BR_UV"] + scaling["baro_stretch_BR_UV"] + 30 - 3.0*S_GRAN) / S_GRAN)); #vec[117] // End impulse response
        SUMZ=0
        for i in range(start,peak):
            a[i]=(i-start)/(peak-start)
            SUMZ=SUMZ+a[i]
        for i in range(peak,end):
            a[i]=(end-i)/(end-peak)
            SUMZ=SUMZ+a[i]
        for i in range(start,end):
            a[i]=a[i]/(SUMZ)
                            
        # Do the same for the cardiopulmonary to venous tone alpha-sympathetic impulse response function.
        start = int(np.round((scaling["baro_delay_CP_UV"] + 5 - 3.0*S_GRAN) / S_GRAN)); #vec[147] // Delay to veins impulse response
        peak  = int(np.round((scaling["baro_delay_CP_UV"] + 9 - 3.0*S_GRAN) / S_GRAN)); #vec[148] // Peak to veins impulse response
        end   = int(np.round((scaling["baro_delay_CP_UV"] + scaling["baro_stretch_CP_UV"] + 40 - 3.0*S_GRAN) / S_GRAN)); #vec[149] // End to veins impulse response
        SUMZ=0
        for i in range(start,peak):
            cpv[i]=(i-start)/(peak-start)
            SUMZ=SUMZ+cpv[i]
        for i in range(peak,end):
            cpv[i]=(end-i)/(end-peak)
            SUMZ=SUMZ+cpv[i]
        for i in range(start,end):
            cpv[i]=cpv[i]/(SUMZ)
                            
        # Do the same for the cardiopulmonary to arterial resistance alpha-sympathetic impulse response function.
        start = int(np.round((scaling["baro_delay_CP_R"] + 2.5 - 3.0*S_GRAN) / S_GRAN)); #vec[150] // Delay to arteries impulse response
        peak  = int(np.round((scaling["baro_delay_CP_R"] + 5.5 - 3.0*S_GRAN) / S_GRAN)); #vec[151] // Peak to arteries impulse response
        end   = int(np.round((scaling["baro_delay_CP_R"] + scaling["baro_stretch_CP_R"] + 35 - 3.0*S_GRAN) / S_GRAN)); #vec[152] // End to arteries impulse response
        SUMZ=0
        for i in range(start,peak):
            cpa[i]=(i-start)/(peak-start)
            SUMZ=SUMZ+cpa[i]
        for i in range(peak,end):
            cpa[i]=(end-i)/(end-peak)
            SUMZ=SUMZ+cpa[i]
        for i in range(start,end):
            cpa[i]=cpa[i]/(SUMZ)
        """
        Assign the parameters the the self function so then can be opened in the main function when called upon
        """
        # Inpulse responses
        self.p = p;
        self.s = s;
        self.a = a;
        self.v =v;
        self.cpv = cpv;
        self.cpa = cpa;
        # Setpoints
        self.ABP_setp = 95;
        self.PP_setp = 35;
        self.RAP_setp = 3; # vec[15]
        self.ABRsc = 18; # vec[1]
        self.RAPsc = 5; # vec[16]
        # Gains
        #self.RRsgain = 0.012; # vec[2]
        #self.RRpgain = 0.009; # vec[3]
        self.RRsgain = 0.012; # vec[2]
        self.RRpgain = 0.005; # vec[3]
        self.rr_sym=0.015; #vec[2] // beta
        self.rr_para=0.09; #vec[3] // para
        self.beta=1;
        self.alpha=1;
        # Timing
        self.S_GRAN = S_GRAN;
        self.SUMZ = SUMZ;
        self.I_length=I_length;
        self.S_INT=S_INT;

