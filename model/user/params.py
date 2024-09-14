from model.user.crossformer import CrossScaleParams, CrossformerParams

encoder_params = {
    "cross_params": [
        CrossScaleParams(
            input_dim=2,
            output_dim=64,
            kernel_size=[4, 8, 16, 32],
            stride=4),
        CrossScaleParams(
            input_dim=64,
            output_dim=128,
            kernel_size=[2, 4],
            stride=2),
        CrossScaleParams(
            input_dim=128,
            output_dim=256,
            kernel_size=[2, 4],
            stride=2),
        CrossScaleParams(
            input_dim=256,
            output_dim=512,
            kernel_size=[2, 4],
            stride=2),
    ],
    "transformer_params": [
        CrossformerParams(
            input_dim=64,
            group=7,
            n_layer=1
        ),
        CrossformerParams(
            input_dim=128,
            group=7,
            n_layer=1
        ),
        CrossformerParams(
            input_dim=256,
            group=7,
            n_layer=3
        ),
        CrossformerParams(
            input_dim=512,
            group=7,
            n_layer=1
        )
    ]
}

decoder_params = {
    "cross_params": [
        CrossScaleParams(
            input_dim=512,
            output_dim=256,
            kernel_size=[2, 4],
            stride=2),
        CrossScaleParams(
            input_dim=256,
            output_dim=128,
            kernel_size=[2, 4],
            stride=2),
        CrossScaleParams(
            input_dim=128,
            output_dim=64,
            kernel_size=[2, 4],
            stride=2),
        CrossScaleParams(
            input_dim=64,
            output_dim=1,
            kernel_size=[4, 8, 16, 32],
            stride=4),
    ],
    "transformer_params": [
        CrossformerParams(
            input_dim=256,
            group=7,
            n_layer=1
        ),
        CrossformerParams(
            input_dim=128,
            group=7,
            n_layer=3
        ),
        CrossformerParams(
            input_dim=64,
            group=7,
            n_layer=1
        ),
        CrossformerParams(
            input_dim=1,
            group=7,
            n_layer=1
        ),
    ]
}
