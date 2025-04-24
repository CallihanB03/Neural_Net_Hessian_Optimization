    # Input Shape 1 x 28 x 28
    cnn_classifier = CNN_classifier(
        input_dim=1,
        hidden_conv_dim1=32,
        hidden_conv_dim2=64,
        hidden_conv_dim3=64,
        hidden_linear_dim1=250,
        hidden_linear_dim2=125,
        hidden_linear_dim3=60,
        output_dim=10,
        kernel_size=3,
        pool_size=2,
        dropout_rate=0.2
    )

    cnn_classifier_summary = summary(cnn_classifier, (1, 28, 28))