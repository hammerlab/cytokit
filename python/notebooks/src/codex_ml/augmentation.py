from imgaug import augmenters as iaa
import imgaug as ia


def get_augmentation_pipeline_01():
    sometimes = lambda aug, p=.5: iaa.Sometimes(p, aug)
    secondary_augmentations = [
        # Noise
        iaa.AdditiveGaussianNoise(scale=(0., 10.)),

        # Randomly alter distribution
        iaa.OneOf([iaa.Multiply((.5, 2.)), iaa.Add((-50, 50))])
    ]

    return iaa.Sequential([

        iaa.Fliplr(.25),
        iaa.Flipud(.25),

        # Crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        ), p=.3),

        # Rotations, scaling and shearing
        sometimes(iaa.Affine(
            rotate=(0., 30.),
            scale={"x": (.8, 1.2), "y": (.8, 1.2)},
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL
        ), p=.3),

        iaa.SomeOf((0, 2), secondary_augmentations)

    ], random_order=False)



