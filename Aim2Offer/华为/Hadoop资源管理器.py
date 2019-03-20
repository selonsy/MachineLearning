#题目描述
#Apache Hadoop YARN是一种新的Hadoop 资源管理器, 主要部件为resource manager和node manager.
#resource manager使用有限状态机维护有状态对象的生命周期.
#RmAppAttempt是resource manager中用于维护一个Application运行尝试的生命周期的数据结构,当某个事件发生时RmAppAttempt会根据当前状态进行状态迁移,同时触发ー个其他行为. 
#RmAppAttempt相关状态和事件简述如下:
#1.RmApp发送start事件创建RmAppAttempt,初始化后,状态迁移为submitted.
#2.RmAppAttempt在submitted状态下收到资源调度器(ResouceScheduler)发来的app_accepted事件,状态迁移为scheduled.
#3.RmAppAttempt在scheduled状态下收到RmContainer发来的container_allocated事件,状态迁移为allocated.
#4.RmAppAttempt在scheduled状态下收到ApplicationMasterLauncher发来的launched事件,状态迁移为running.
#5.RmAppAttempt在running状志下收到ResouceScheduler发来的finished事件,状志迁移为finished.
#6.在RmAppAttempt运行过程中(不含finished状态),当收到来自客户端的杀死应用程序的命令时,RmApp向RmAppAttempt发送kill事件,RmAppattemp收到后,状态迁移到killed状态.
#请编写一段程序模拟实现RmAppAttempt状态机,接收一段连续事件输入,输出每个事件处理后yarn中对应RmAppAttempt对象的状态.
#输入描述:
#模拟RmAppAttempt状态机,接收一段连续事件输入
#输入接口:(事件间用空格分隔)
#事件源|RmAppAtcempt对象事件
#事件源:RmApp,Resoucescheduler,ApplicationMasterLauncher,RmContainer
#事件: start, app_accepted,container_allocated,launched, finished,kill
#输出描述:
#输出接口:(输出结果间用分号;分隔)
#RmAppAtcenpt对象|状态
#状态:submitted,scheduled,allocated,running,finished,killed
#无效输入通过状态机过滤, 不需要任何输出

#输入
#RmApp|RmAppAttempt_001|start Rmapp|RmAppAttempt_002|start RmApp|RmappAttempt_001|kill
#输出
#RmAppAttempt_001|submitted;RmAppAttempt_002|submitted;RmAppAttempt_001|killed

