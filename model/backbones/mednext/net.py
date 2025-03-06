import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from model.backbones.mednext.blocks import *
from model.backbones.generic import Seg3D
from model.backbones.registry import backbone_registry


class MedNeXt(Seg3D):

    def __init__(
        self,
        input_shape: list[int],
        output_shape: list[int],
        n_channels: int = 32,
        exp_r: list[int] = [
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ],  # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,  # Ofcourse can test kernel_size
        do_res: bool = False,  # Can be used to individually test residual connection
        do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
        checkpoint_style: bool = None,  # Either inside block or outside block
        block_counts: list = [
            2,  # encoder 0
            2,  # encoder 1
            2,  # encoder 2
            2,  # encoder 3
            2,  # bottleneck
            2,  # decoder 0
            2,  # decoder 0
            2,  # decoder 0
            2,  # decoder 0
        ],  # Can be used to test staging ratio:
        # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type="group",
        dim="3d",  # 2d or 3d
        grn=False,
        loss_args={
            "name": "DiceCELossNNUNET",
        },
    ):

        super().__init__(input_shape, output_shape, loss_args=loss_args)

        assert checkpoint_style in [None, "outside_block"]
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == "outside_block":
            self.outside_block_checkpointing = True
        assert dim in ["2d", "3d"]

        self.encoder = Encoder(
            in_channels=input_shape[0],
            n_channels=n_channels,
            exp_r=exp_r[: len(block_counts) // 2 + 1],
            enc_kernel_size=kernel_size,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=block_counts[: len(block_counts) // 2 + 1],
            norm_type=norm_type,
            dim=dim,
            grn=False,
            outside_block_checkpointing=self.outside_block_checkpointing,
        )

        self.decoder = Decoder(
            n_classes=output_shape[0],
            n_channels=n_channels,
            exp_r=exp_r[len(block_counts) // 2 + 1 :],
            kernel_size=kernel_size,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=block_counts[len(block_counts) // 2 + 1 :],
            norm_type=norm_type,
            dim=dim,
            grn=False,
            outside_block_checkpointing=self.outside_block_checkpointing,
        )


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        n_channels: int = 32,
        exp_r: list[int] = [
            4,
            4,
            4,
            4,
            4,
        ],
        enc_kernel_size: int = 7,
        do_res: bool = False,
        do_res_up_down: bool = False,
        block_counts: list = [
            2,  # encoder 0
            2,  # encoder 1
            2,  # encoder 2
            2,  # encoder 3
            2,  # bottleneck
        ],
        norm_type="group",
        dim="3d",
        grn=False,
        outside_block_checkpointing=False,
    ):

        super().__init__()

        self.outside_block_checkpointing = outside_block_checkpointing

        if dim == "2d":
            conv = nn.Conv2d
        elif dim == "3d":
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=1)

        self.enc_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    exp_r=exp_r[0],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[0])
            ]
        )

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.enc_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 2,
                    out_channels=n_channels * 2,
                    exp_r=exp_r[1],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[1])
            ]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 4,
                    out_channels=n_channels * 4,
                    exp_r=exp_r[2],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[2])
            ]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 8,
                    out_channels=n_channels * 8,
                    exp_r=exp_r[3],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[3])
            ]
        )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 16,
                    out_channels=n_channels * 16,
                    exp_r=exp_r[4],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[4])
            ]
        )

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):

        x = self.stem(x)

        features = []
        if self.outside_block_checkpointing:

            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            features.append(x_res_0)

            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            features.append(x_res_1)

            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            features.append(x_res_2)

            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)
            features.append(x_res_3)

            x = self.iterative_checkpoint(self.bottleneck, x)
            features.append(x)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            features.append(x_res_0)

            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            features.append(x_res_1)

            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            features.append(x_res_2)

            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)
            features.append(x_res_3)

            x = self.bottleneck(x)
            features.append(x)

        return features


class Decoder(nn.Module):

    def __init__(
        self,
        n_classes: int,
        n_channels: int = 32,
        exp_r: list[int] = 4,
        kernel_size: int = 7,
        dec_kernel_size: int = None,
        do_res: bool = False,
        do_res_up_down: bool = False,
        checkpoint_style: bool = None,
        block_counts: list = [
            2,  # decoder 0
            2,  # decoder 1
            2,  # decoder 2
            2,  # decoder 3
        ],
        norm_type="group",
        dim="3d",
        grn=False,
        outside_block_checkpointing=False,
    ):

        super().__init__()

        self.outside_block_checkpointing = outside_block_checkpointing

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[0],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 8,
                    out_channels=n_channels * 8,
                    exp_r=exp_r[0],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[0])
            ]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[1],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 4,
                    out_channels=n_channels * 4,
                    exp_r=exp_r[1],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[1])
            ]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[2],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 2,
                    out_channels=n_channels * 2,
                    exp_r=exp_r[2],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[2])
            ]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[3],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    exp_r=exp_r[3],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[3])
            ]
        )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, features):

        x_res_0, x_res_1, x_res_2, x_res_3, x = features

        if self.outside_block_checkpointing:

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            # del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            # del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            # del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            # del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)
            # del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            # del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            # del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            # del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        return [*features, x_up_3, x_up_2, x_up_1, x_up_0, x]


@backbone_registry.register
class MedNeXtSmall(MedNeXt):
    def __init__(
        self,
        input_shape,
        output_shape,
        *args,
        **kwargs,
    ):

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            n_channels=32,
            exp_r=[4, 4, 4, 4, 4, 4, 4, 4, 4],
            kernel_size=3,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            *args,
            **kwargs,
        )


@backbone_registry.register
class MedNeXtMedium(MedNeXt):
    def __init__(
        self,
        input_shape,
        output_shape,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_shape,
            output_shape,
            n_channels=32,
            exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
            kernel_size=3,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
            checkpoint_style="outside_block",
            *args,
            **kwargs,
        )


@backbone_registry.register
class MedNeXtLarge(MedNeXt):
    def __init__(
        self,
        input_shape,
        output_shape,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_shape,
            output_shape,
            n_channels=32,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=3,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style="outside_block",
            *args,
            **kwargs,
        )


if __name__ == "__main__":

    from pprint import pprint

    TEST_ENCODER = False
    TEST_DECODER = False
    TEST_NET = True

    device = "cuda:4"

    if TEST_ENCODER:

        x = torch.randn(1, 3, 128, 128, 128).to(device)
        encoder = Encoder(in_channels=3).to(device)
        features = encoder(x)

        print("works!")

    if TEST_DECODER:

        x = torch.randn(1, 3, 128, 128, 128).to(device)

        encoder = Encoder(in_channels=3).to(device)
        decoder = Decoder(n_classes=10).to(device)
        features = encoder(x)
        out = decoder(features)

        print("works!")

    if TEST_NET:

        # net = MedNeXtSmall(
        #     input_shape=[3, 128, 128, 128],
        #     output_shape=[5, 128, 128, 128],
        #     loss_args={
        #         "name": "RCELoss",
        #     },
        # ).to(device)
        # net.unit_test()
        # pprint(net.feature_shapes)
        # del net

        net = MedNeXtMedium(
            input_shape=[3, 128, 128, 128],
            output_shape=[5, 128, 128, 128],
            loss_args={
                "name": "RCELoss",
            },
        ).to(device)
        net.unit_test()
        pprint(net.feature_shapes)
        del net

        # net = MedNeXtLarge(
        #     input_shape=[3, 128, 128, 128],
        #     output_shape=[5, 128, 128, 128],
        #     loss_args={
        #         "name": "RCELoss",
        #     },
        # ).to(device)
        # net.unit_test()
        # pprint(net.feature_shapes)
        # del net
