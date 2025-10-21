# export HF_TOKEN=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


# wget --header="Authorization: Bearer $HF_TOKEN" \
#     https://huggingface.co/datasets/Raymond-Qiancx/COCO/resolve/main/images/h5_tienkung_xsens_1rgb.tar.gz

# tar -xzvf /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/images/h5_tienkung_xsens_1rgb.tar.gz \
#     -C /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/images/

# wget --header="Authorization: Bearer $HF_TOKEN" \
#     https://huggingface.co/datasets/Raymond-Qiancx/COCO/resolve/main/images/h5_tienkung_xsens_1rgb.tar.gz

for f in /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/images/*.tar.gz; do
    tar -xzf "$f" -C /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/images
done
