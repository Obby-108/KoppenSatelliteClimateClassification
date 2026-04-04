import timm

def get_timm_sentinel_swin(num_classes=30):
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=True,
        in_chans=12,
        num_classes=num_classes
    )

    return model
