import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid, relu
from tensorflow.keras.layers import Input, Conv2D, Multiply, Add, LayerNormalization, Lambda, LeakyReLU, BatchNormalization


def BSConv(inputs, out_channels, kernel_size=3, strides=1, padding='SAME', dilation=1, bias=True):
    # conv 1X1
    x = Conv2D(out_channels, 1, strides=1, padding='SAME')(inputs) 

    # depth-wise conv
    x = Conv2D(out_channels,
               kernel_size,
               strides=strides,
               padding=padding,
               dilation_rate=dilation,
               groups=out_channels,
               use_bias=bias)(x)
    return x  


def PartialConv(inputs, in_channels, n_div=2):
    # channel dimension split according to ratio i.e. n_div
    dim_conv = in_channels // n_div
    dim_untouched = in_channels - dim_conv

    # aggregate high SR performance contribution channels to the former part through fully connected layer/conv 1X1
    x = Conv2D(in_channels, 1, strides=1, padding='SAME')(inputs)

    # feature maps split according to calculated dim_conv/dim_untouched
    x1, x2 = tf.split(x, [dim_conv, dim_untouched], axis=-1)

    # large receptive filed partial depth-wise convolution
    x1 = Conv2D(dim_conv, 5, strides=1, padding='SAME')(x1)
    x1 = Conv2D(dim_conv, 5, strides=1, padding='SAME', dilation_rate=3)(x1)

    # concate
    x = tf.concat([x1,x2], axis=-1)

    return x


def PFE(inputs, in_channels, n_div=2, mix_ratio=2):
    'Partial feature extraction'
    # partial conv followed by pwconv channel mixing
    x = PartialConv(inputs, in_channels, n_div=n_div)
    x = Conv2D(in_channels*mix_ratio, kernel_size=1, padding='SAME')(x)
    x = BatchNormalization(axis=-1)(x)
    x = relu(x)
    x = Conv2D(in_channels, kernel_size=1, padding='SAME')(x)

    return x


def PPA(inputs, in_channels, n_div=2):
    'Partial pixel attention'
    # channel dimension split according to ratio i.e. n_div
    dim_conv = in_channels // n_div
    dim_untouched = in_channels - dim_conv

    # aggregate high SR performance contribution channels to the former part through fully connected layer/conv 1X1
    x = Conv2D(in_channels, 1, strides=1, padding='SAME')(inputs)

    # feature maps split according to calculated dim_conv/dim_untouched
    x1, x2 = tf.split(x, [dim_conv, dim_untouched], axis=-1)

    # partial pixel attention(i.e. conduct pixel attention on x1)
    # if replace "tf.keras.activations.relu(x)" with "relu(x)" which is loaded by "from tensorflow.keras.activations import sigmoid, relu", convert_to_tflite procedure reports bug
    attn = Lambda(lambda x: tf.keras.activations.relu(x))(x1)
    attn = Conv2D(dim_conv, 3, strides=1, padding='SAME', activation='sigmoid')(attn)
    x1 = Multiply()([attn, x1])

    # concate
    x = tf.concat([x1,x2], axis=-1)

    return x


def MPFD(inputs, in_channels=16):
    'Multi-stage partial feature distillation'
    out_1 = PFE(inputs, in_channels)
    out_2 = PFE(out_1, in_channels)
    out_3 = PFE(out_2, in_channels)

    # channel reduction before feature maps concate
    refine_1 = Conv2D(in_channels//4, 1, strides=1, padding='SAME')(out_1)
    refine_2 = Conv2D(in_channels//4, 1, strides=1, padding='SAME')(out_2)
    refine_3 = Conv2D(in_channels//4, 1, strides=1, padding='SAME')(out_3)
    refine_inputs = Conv2D(in_channels//4, 1, strides=1, padding='SAME')(inputs)

    # feature maps concate and fuse
    out = tf.concat((refine_1, refine_2, refine_3, refine_inputs), axis=-1) # concate(4:4:4:4) -> 16

    # Partial pixel attention
    out = PPA(out, in_channels)
     
    return out
    

def AFA(x, ref, dim=3):
    'Attention-based frame alignment'
    k_conv = Conv2D(dim, 3, strides=1, padding='SAME')
    v_conv = Conv2D(dim, 3, strides=1, padding='SAME')
    q_conv = Conv2D(dim, 3, strides=1, padding='SAME')

    # deprecate inputs LayerNormalization -> incur bug when converting to tflite
    # x = LayerNormalization()(x)
    # ref = LayerNormalization()(ref)

    # Linear project
    q = q_conv(ref)
    k = k_conv(x)
    v = v_conv(x)

    # Reshape
    b , h, w, c = tf.shape(q)
    q = tf.reshape(q, [b, h * w, c])
    k = tf.reshape(k, [b, h * w, c])
    k = tf.transpose(k, perm=[0, 2, 1])
    v = tf.reshape(v, [b, h * w, c])
    
    # Calculate attention weights
    attn_scores = tf.matmul(k, q)
    attn_scores = attn_scores / tf.sqrt(tf.cast(c, tf.float32))
    att_weights = tf.nn.softmax(attn_scores, axis=-1)

    # Execute attention/weighted sum
    y = tf.matmul(v, att_weights)

    # Reshape
    y = tf.reshape(y, [b, h, w, c])

    # Residual connection/shortcut
    y = Add()([x, y])

    return y


def ConvTail(inputs, num_out_ch, kernel_size=3, strides=1, padding='SAME', bias=False, groups=1, dilation=1, scale=4):
    outputs =  Conv2D(num_out_ch*scale*scale,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=bias,
                      dilation_rate=dilation,)(inputs)
    return outputs


def FunctionalFANI(inputs, num_out_ch=3, scale=4, num_feat=16, num_blocks=4):
    x_pre = inputs[:, :, :, :3]
    x = inputs[:, :, :, 3:6]
    x_nex = inputs[:, :, :, 6:]
    shape = tf.shape(x)

    # construct auxiliary frame 
    ref = tf.concat([x_pre, x_nex], axis=-1)
    ref = Conv2D(1, 3, strides=1, padding='SAME')(ref)

    # frame alignment/auxiliary information propagation
    x_forward = AFA(x, ref)

    # shallow feature extraction
    x_forward = BSConv(x_forward, num_feat)

    # deep feature extraction
    for i in range(num_blocks):
        x_forward = MPFD(x_forward, in_channels=num_feat)

    # increase channel dimension prepare for pixel shuffle
    out = ConvTail(x_forward, num_out_ch, scale=scale)

    # pixel shuffle
    out = tf.nn.depth_to_space(out, scale)
    
    # bilinear shortcut and pixel value clip
    bilinear = tf.image.resize(x, size=(shape[1] * scale, shape[2] * scale))
    out = Add()([out, bilinear])
    out = tf.clip_by_value(out, 0, 255)

    return out


def FANI():
    x = Input(shape=(None, None, 9))
    return Model(inputs=x, outputs=FunctionalFANI(x))