import os

if(os.environ.get('PLATFORM') == "CUB"):
    IMAGES_CLASS_MOBILENET_V3_SMALL_0 = [
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

    IMAGES_CLASS_MOBILENET_V3_SMALL_1 = [
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

    IMAGES_CLASS_MOBILENET_V3_SMALL_2 = [
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



#MOBILENET_V3_SMALL_LAYERS = ['features.21', 'features.24', 'features.26']
MOBILENET_V3_SMALL_LAYERS = ['features.5', 'features.6', 'features.7']
lambda_mobilenet_v3_small = [0.5]
#lambda_mobilenet_v3_small = [0.6,0.7,0.9,1]
