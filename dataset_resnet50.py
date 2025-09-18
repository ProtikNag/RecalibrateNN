import os

if(os.environ.get('PLATFORM') == "Srikanth"):
    IMAGES_CLASS_RESNET50_0 = ['/home/multiclass_classification/deer/train/61b3d45750.jpg',
    '/home/multiclass_classification/deer/train/image_1.jpg',
    '/home/multiclass_classification/deer/train/ad480eded9.jpg',
    '/home/multiclass_classification/deer/train/93769d3783.jpg',
    '/home/multiclass_classification/deer/train/a2ffbd370e.jpg',
    '/home/multiclass_classification/deer/train/958c145e72.jpg',
    '/home/multiclass_classification/deer/train/5e0833a1e6.jpg',
    '/home/multiclass_classification/deer/train/1d5488fd16.jpg',
    '/home/multiclass_classification/deer/train/04a53a216b.jpg',
    '/home/multiclass_classification/deer/train/7bf91daa47.jpg',
     ]

    IMAGES_CLASS_RESNET50_1 = ['/home/multiclass_classification/horse/train/02_047.png',
    '/home/multiclass_classification/horse/train/07_009.png',
    '/home/multiclass_classification/horse/train/07_034.png',
    '/home/multiclass_classification/horse/train/02_102.png',
    '/home/multiclass_classification/horse/train/02_037.png',
    '/home/multiclass_classification/horse/train/02_092.png',
    '/home/multiclass_classification/horse/train/03_055.png',
    '/home/multiclass_classification/horse/train/06_121.png',
    '/home/multiclass_classification/horse/train/02_062.png',
    '/home/multiclass_classification/horse/train/02_063.png'
    ]
    
    IMAGES_CLASS_RESNET50_2 = ['/home/multiclass_classification/zebra/train/n02391049_686.jpg',
    '/home/multiclass_classification/zebra/train/045.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_11041.jpg',
    '/home/multiclass_classification/zebra/train/070.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_2177.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_1199.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_10454.jpg',
    '/home/multiclass_classification/zebra/train/image_135.jpeg',
    '/home/multiclass_classification/zebra/train/072.jpg',
    '/home/multiclass_classification/zebra/train/046.jpg' ]

if(os.environ.get('PLATFORM') == "CUB"):
    IMAGES_CLASS_RESNET50_0 = [
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

    IMAGES_CLASS_RESNET50_1 = [
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

    IMAGES_CLASS_RESNET50_2 = [
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
RESNET50_LAYERS = ['layer1.0.conv2','layer1.0.conv3','layer1.1.conv2','layer1.2.conv1','layer1.2.conv2','layer2.0.conv1',
'layer2.0.conv3','layer2.0.downsample.0','layer2.1.conv1','layer2.1.conv2','layer2.1.conv3','layer2.2.conv1',
'layer2.2.conv2','layer2.3.conv1','layer2.3.conv2','layer2.3.conv3','layer3.0.conv1','layer3.0.conv2','layer3.0.conv3',
'layer3.0.downsample.0','layer3.1.conv1','layer3.1.conv2','layer3.2.conv1','layer3.2.conv2','layer3.2.conv3',
'layer3.3.conv1','layer3.3.conv2','layer3.3.conv3','layer3.4.conv1','layer3.4.conv2','layer3.4.conv3','layer3.5.conv1',
'layer3.5.conv2','layer3.5.conv3','layer4.0.conv1','layer4.0.conv3','layer4.0.downsample.0','layer4.1.conv1',
'layer4.1.conv2','layer4.2.conv2']
"""
RESNET50_LAYERS = ['layer1.0.conv2','layer1.0.conv3','layer4.0.conv1','layer4.0.conv2'] 

lambda_resnet50 = [0.5]
#lambda_resnet50 = [0.6,0.7,0.9,1]
