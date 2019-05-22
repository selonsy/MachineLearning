a = {'me': 493945, 'other': 1159849}
b = {'me': {'success': 380, 'fail': 198}, 'other': {'success': 987, 'fail': 585}}

total_time = a["me"]+a["other"]
total_count = b['me']['success'] + b['me']['fail'] + b['other']['success'] +b['other']['fail']
total_count_success = b['me']['success'] + b['other']['success']
total_count_fail = b['me']['fail'] + b['other']['fail']
print("语音通话总时长：{0} 秒，也即为：{1} 分钟，{2} 小时，{3} 天".format(
   total_time, total_time // 60, total_time // 3660, total_time // (3600*24)
))
print("语音通话总次数：{} 次，其中成功 {} 次，占比 {} %；失败 {} 次，占比 {} %".format(
   total_count, total_count_success, round(total_count_success*100/total_count), 
   total_count_fail, round(total_count_fail*100/total_count)
))
print("其中，彦承累计拨出通话时长 {} 秒，占比 {} %；邱小宝累计拨出通话时长 {} 秒，占比 {} %".format(
    a['me'], round(a['me']*100 / total_time,2),a['other'],round(a['other']*100 / total_time,2)
))
print("彦承累计拨出通话次数 {} 次，占比 {} %；其中成功 {} 次，失败 {} 次".format(
    b['me']['success'] + b['me']['fail'], round((b['me']['success'] + b['me']['fail'])*100 /total_count,2), b['me']['success'],b['me']['fail']
))
print("邱小宝累计拨出通话次数 {} 次，占比 {} %；其中成功 {} 次，失败 {} 次".format(
    b['other']['success'] + b['other']['fail'],  round((b['other']['success'] + b['other']['fail'])*100 /total_count,2),b['other']['success'],b['other']['fail']
))