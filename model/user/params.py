from model.user.crossformer import CrossScaleParams, CrossformerParams

encoder_params = {
    "cross_params": [
        CrossScaleParams(
            input_dim=2,
            output_dim=96,
            kernel_size=[4, 8, 16, 32],
            stride=4),
        CrossScaleParams(
            input_dim=96,
            output_dim=192,
            kernel_size=[2, 4],
            stride=2),
        CrossScaleParams(
            input_dim=192,
            output_dim=384,
            kernel_size=[2, 4],
            stride=2),
        CrossScaleParams(
            input_dim=384,
            output_dim=768,
            kernel_size=[2, 4],
            stride=2),
    ],
    "transformer_params": [
        CrossformerParams(
            input_dim=96,
            group=7,
            n_layer=1
        ),
        CrossformerParams(
            input_dim=192,
            group=7,
            n_layer=1
        ),
        CrossformerParams(
            input_dim=384,
            group=7,
            n_layer=9
        ),
        CrossformerParams(
            input_dim=768,
            group=7,
            n_layer=1
        )
    ]
}

decoder_params = {
    "cross_params": [
        CrossScaleParams(
            input_dim=768,
            output_dim=384,
            kernel_size=[2, 4],
            stride=2),
        CrossScaleParams(
            input_dim=384,
            output_dim=192,
            kernel_size=[2, 4],
            stride=2),
        CrossScaleParams(
            input_dim=192,
            output_dim=96,
            kernel_size=[2, 4],
            stride=2),
        CrossScaleParams(
            input_dim=96,
            output_dim=1,
            kernel_size=[4, 8, 16, 32],
            stride=4),
    ],
    "transformer_params": [
        CrossformerParams(
            input_dim=768,
            group=7,
            n_layer=1
        ),
        CrossformerParams(
            input_dim=384,
            group=7,
            n_layer=9
        ),
        CrossformerParams(
            input_dim=192,
            group=7,
            n_layer=1
        ),
        CrossformerParams(
            input_dim=96,
            group=7,
            n_layer=1
        ),
    ]
}
