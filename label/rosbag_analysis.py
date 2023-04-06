import rosbag
bag_file = "/home/qiu/2023-01-07-13-23-38_3.bag"
bag =rosbag.Bag(bag_file,"r")
info = bag.get_type_and_topic_info()
print(info)