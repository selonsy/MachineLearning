from train.run_Train_SiamFPN import train

if __name__ == "__main__":
    
    # data_dir = "/home/hfan/Dataset/ILSVRC2015_crops/Data/VID/train"
    # train_imdb = "/home/hfan/Desktop/PyTorch-SiamFC/ILSVRC15-curation/imdb_video_train.json"
    # val_imdb = "/home/hfan/Desktop/PyTorch-SiamFC/ILSVRC15-curation/imdb_video_val.json"

    # Windows ILSVRC15
    data_dir = r"D:\workspace\MachineLearning\asimo\ILSVRC_crops\Data\VID\train"
    train_imdb = r"D:\workspace\MachineLearning\asimo\train\ILSVRC15-curation\imdb_video_train.json"
    val_imdb = r"D:\workspace\MachineLearning\asimo\train\ILSVRC15-curation\imdb_video_val.json"

    # Windows OTB
    data_dir = r"D:\workspace\MachineLearning\asimo\OTB_train_crops\img"
    train_imdb = r"D:\workspace\MachineLearning\asimo\imdb_video_train_otb.json"
    val_imdb = r"D:\workspace\MachineLearning\asimo\imdb_video_val_otb.json"

    # Linux    
    # data_dir = r"/home/sjl/vot/SiamFPN/ILSVRC_crops/Data/VID/train"
    # train_imdb = r"/home/sjl/vot/SiamFPN/train/ILSVRC15-curation/imdb_video_train.json"
    # val_imdb = r"/home/sjl/vot/SiamFPN/train/ILSVRC15-curation/imdb_video_val.json"

    # Linux OTB
    # data_dir = r"/home/sjl/vot/SiamFPN/OTB_train_crops/img"
    # train_imdb = r"/home/sjl/vot/SiamFPN/imdb_video_train_otb.json"
    # val_imdb = r"/home/sjl/vot/SiamFPN/imdb_video_val_otb.json"

    # training SiamFC network, using GPU by default
    train(data_dir, train_imdb, val_imdb)
