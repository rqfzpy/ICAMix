from .vit import ViT
from .swin import SwinTransformer
from .resnet import model_dict

def create_model(img_size, n_classes, args):
    if args.model == 'vit':
        if img_size == 32:
            patch_size = 2
        elif img_size == 64:
            patch_size = 4
        else:
            patch_size = 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12)
        
    elif args.model =='swin':

        depths = [2, 6,4]
        num_heads = [2, 4,8]
        mlp_ratio = 2
        window_size = 4

        if img_size == 32:
            patch_size = 2
        elif img_size == 64:
            patch_size = 4
        else:
            patch_size = 8
        window_size = patch_size*2

        model = SwinTransformer(img_size=img_size, window_size=window_size, 
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes)

    elif 'resnet' in args.model:
        resnet_constructor = model_dict[args.model][0]  
        model = resnet_constructor(num_classes=n_classes)  

    return model