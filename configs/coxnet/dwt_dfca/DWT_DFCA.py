_base_ = '../coxnet_r50_fpn_1x_rgbtdroneperson.py'

model = dict(use_clfm=['dwt_dfca'])

work_dir = 'work_dir/coxnet/rgbtdroneperson/dwt_dfca_oneway'
