import os
#Mixed_6b.branch7x7dbl_4.conv_0.6
if(os.environ.get('PLATFORM') == "Srikanth"):
    IMAGES_CLASS_INCEPTION_V3_0 = ['/home/multiclass_classification/deer/train/8e2539f451.jpg',
    '/home/multiclass_classification/deer/train/40a63aaa33.jpg',
    '/home/multiclass_classification/deer/train/i-2022-04-05T150314-633_jpg.rf.dec4fad9135f9f4e7e32cecb57c74808.jpg',
    '/home/multiclass_classification/deer/train/1c345fa7c8.jpg',
    '/home/multiclass_classification/deer/train/789d5733d0.jpg',
    '/home/multiclass_classification/deer/train/image_10.jpg',
    '/home/multiclass_classification/deer/train/ea619162ac.jpg',
    '/home/multiclass_classification/deer/train/9d1073d109.jpg',
    '/home/multiclass_classification/deer/train/6435dd1e94.jpg',
    '/home/multiclass_classification/deer/train/9b1e8519de2.jpg'
     ]
    IMAGES_CLASS_INCEPTION_V3_1 = ['/home/multiclass_classification/horse/train/05_002.png',
    '/home/multiclass_classification/horse/train/03_045.png',
    '/home/multiclass_classification/horse/train/03_092.png',
    '/home/multiclass_classification/horse/train/07_067.png',
    '/home/multiclass_classification/horse/train/03_091.png',
    '/home/multiclass_classification/horse/train/06_088.png',
    '/home/multiclass_classification/horse/train/06_041.png',
    '/home/multiclass_classification/horse/train/horse02-3.png',
    '/home/multiclass_classification/horse/train/03_086.png',
    '/home/multiclass_classification/horse/train/horse22-6.png'
    ]
    IMAGES_CLASS_INCEPTION_V3_2 = [ '/home/multiclass_classification/zebra/train/image_31.jpeg',
    '/home/multiclass_classification/zebra/train/n02391049_2162.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_5367.jpg',
    '/home/multiclass_classification/zebra/train/025.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_3262.jpg',
    '/home/multiclass_classification/zebra/train/009.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_10158.jpg',
    '/home/multiclass_classification/zebra/train/image_36.jpeg',
    '/home/multiclass_classification/zebra/train/n02391049_8008.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_2803.jpg'
    ]

if(os.environ.get('PLATFORM') == "CUB"):
    IMAGES_CLASS_INCEPTION_V3_0 = [
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0009_34.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0025_796057.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0036_796127.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0046_18.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0058_796074.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0079_796122.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0016_796067.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0035_796140.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0039_796132.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0049_796063.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0069_796139.jpg',
    '/mnt/sdd/caltech_training/recalib/001.Black_footed_Albatross/valid/Black_Footed_Albatross_0086_796062.jpg'
    ]

    IMAGES_CLASS_INCEPTION_V3_1 = [
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0050_43084.jpg',
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0053_43220.jpg',
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0069_43541.jpg',
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0081_42779.jpg',
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0089_43013.jpg',    
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0097_43567.jpg',
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0051_43043.jpg',
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0058_43251.jpg',
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0070_43516.jpg',
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0082_42989.jpg',    
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0091_43066.jpg',
    '/mnt/sdd/caltech_training/recalib/044.Frigatebird/valid/Frigatebird_0114_42807.jpg'
    
    ]

    IMAGES_CLASS_INCEPTION_V3_2 = [
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0003_94427.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0040_94051.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0059_94504.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0074_93692.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0111_93872.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0127_93700.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0025_95218.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0056_95229.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0070_93678.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0075_95357.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0118_93475.jpg',
    '/mnt/sdd/caltech_training/recalib/100.Brown_Pelican/valid/Brown_Pelican_0139_93995.jpg'
    ]

if(os.environ.get('PLATFORM') == "INET"):
    pass
        
"""
INCEPTION_V3_LAYERS = ['Mixed_6b.branch7x7dbl_4.conv', 'Mixed_6c.branch1x1.conv',
'Mixed_6c.branch7x7_2.conv', 'Mixed_6c.branch7x7_3.conv', 'Mixed_6c.branch7x7dbl_1.conv',
'Mixed_6c.branch7x7dbl_2.conv', 'Mixed_6c.branch_pool.conv', 'Mixed_6d.branch7x7dbl_1.conv',
'Mixed_6d.branch7x7dbl_2.conv', 'Mixed_6d.branch7x7dbl_4.conv', 'Mixed_6e.branch1x1.conv',
'Mixed_6e.branch7x7_1.conv', 'Mixed_6e.branch7x7_2.conv', 'Mixed_6e.branch7x7dbl_2.conv',
'Mixed_6e.branch7x7dbl_5.conv', 'Mixed_7a.branch3x3_2.conv', 'Mixed_7a.branch7x7x3_1.conv',
'Mixed_7b.branch3x3_1.conv', 'Mixed_7b.branch3x3dbl_1.conv', 'Mixed_7b.branch3x3dbl_2.conv']


INCEPTION_V3_LAYERS = ['Mixed_6b.branch7x7dbl_4.conv', 'Mixed_6c.branch1x1.conv',
'Mixed_6c.branch7x7_2.conv', 'Mixed_6c.branch7x7_3.conv', 'Mixed_6c.branch7x7dbl_1.conv',
'Mixed_6c.branch7x7dbl_2.conv', 'Mixed_6c.branch_pool.conv', 'Mixed_6d.branch7x7dbl_1.conv',
'Mixed_6d.branch7x7dbl_2.conv', 'Mixed_6d.branch7x7dbl_4.conv', 'Mixed_6e.branch1x1.conv' ] 
"""

INCEPTION_V3_LAYERS = ['Mixed_6e.branch1x1.conv']
lambda_inception = [0.5]
#lambda_inception = [0.6,0.8]
