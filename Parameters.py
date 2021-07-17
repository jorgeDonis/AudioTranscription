Parameters = {
    'STFT' : 
    {
        'HOP_LENGTH'    :   128,
        'WIN_LENGTH'    :   1024,
        'N_FFT'         :   1024,
    },
    'SPEC' :
    {
        'IMG_HEIGHT'    :   192,
        'V_MIN'         :   -70, #minimmum pixel intensity
        'V_MAX'         :   8,
        'F_MIN'         :   27.5,
        'F_MAX'         :   3520
    },
    'TRAINING':
    {
        'EPOCHS'        :   40,
        'BATCH_SIZE'    :   2,
        'POOLING_RATIO' :   8 # n^(number of poolings), where n is the size of the pooling in the x axis
    }
}