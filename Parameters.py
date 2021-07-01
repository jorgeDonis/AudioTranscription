Parameters = {
    'STFT' : 
    {
        'HOP_LENGTH' : 512,
        'WIN_LENGTH' : 2048
    },
    'SPEC' :
    {
        'IMG_WIDTH_PER_FRAME'   :   1       ,   #integer value
        'IMG_HEIGHT'            :   256     ,
        'Y_AXIS_SCALE'          :   'linear'    #linear or log
    },
    'TRAINING':
    {
        'EPOCHS' : 20,
        'BATCH_SIZE' : 7,
        'POOLING_RATIO' : 8 # n^(number of poolings), where n is the size of the pooling in the x axis
    }
}