#Python协程：从yield/send到async/await

#当一个函数中包含yield语句时，python会自动将其识别为一个生成器。这时fib(20)并不会真正调用函数体，而是以函数体生成了一个生成器对象实例。
#yield在这里可以保留fib函数的计算现场，暂停fib的计算并将b返回。而将fib放入for…in循环中时，每次循环都会调用next(fib(20))，唤醒生成器，执行到下一个yield语句处，直到抛出StopIteration异常。此异常会被for循环捕获，导致跳出循环。

# def fib(n):
#     index = 0
#     a = 0
#     b = 1
#     while index < n:
#         yield b
#         a, b = b, a + b
#         index += 1
# print('-'*10 + 'test yield fib' + '-'*10)
# for fib_res in fib(20):
# 	print(fib_res)


# 其中next(sfib)相当于sfib.send(None)，可以使得sfib运行至第一个yield处返回。后续的sfib.send(random.uniform(0, 0.5))则将一个随机的秒数发送给sfib，作为当前中断的yield表达式的返回值。这样，我们可以从“主”程序中控制协程计算斐波那契数列时的思考时间，协程可以返回给“主”程序计算结果，Perfect！
# import time,random
# def stupid_fib(n):
# 	index = 0
# 	a = 0
# 	b = 1
# 	while index < n:
# 		sleep_cnt = yield b
# 		print('let me think {0} secs'.format(sleep_cnt))
# 		time.sleep(sleep_cnt)
# 		a, b = b, a + b
# 		index += 1
# print('-'*10 + 'test yield send' + '-'*10)
# N = 20
# sfib = stupid_fib(N)
# fib_res = next(sfib)
# while True:
# 	print(fib_res)
# 	try:
# 		fib_res = sfib.send(random.uniform(0, 0.5))
# 	except StopIteration:
# 		break

# async def hello():
#     print("Hello world!")
#     r = await asyncio.sleep(1)
#     print("Hello again!")

#重构生成器
# def copy_fib(n):
# 	print('I am copy from fib')
# 	yield from fib(n)
# 	print('Copy end')
# print('-'*10 + 'test yield from' + '-'*10)
# for fib_res in copy_fib(20):
# 	print(fib_res)


# def copy_stupid_fib(n):
# 	print('I am copy from stupid fib')
# 	yield from stupid_fib(n)
# 	print('Copy end')
# print('-'*10 + 'test yield from and send' + '-'*10)
# N = 20
# csfib = copy_stupid_fib(N)
# fib_res = next(csfib)
# while True:
# 	print(fib_res)
# 	try:
# 		fib_res = csfib.send(random.uniform(0, 0.5))
# 	except StopIteration:
# 		break

import asyncio,random
# @asyncio.coroutine
# def smart_fib(n):
# 	index = 0
# 	a = 0
# 	b = 1
# 	while index < n:
# 		sleep_secs = random.uniform(0, 0.2)
# 		yield from asyncio.sleep(sleep_secs)
# 		print('Smart one think {} secs to get {}'.format(sleep_secs, b))
# 		a, b = b, a + b
# 		index += 1
 
# @asyncio.coroutine
# def stupid_fib(n):
# 	index = 0
# 	a = 0
# 	b = 1
# 	while index < n:
# 		sleep_secs = random.uniform(0, 0.4)
# 		yield from asyncio.sleep(sleep_secs)
# 		print('Stupid one think {} secs to get {}'.format(sleep_secs, b))
# 		a, b = b, a + b
# 		index += 1
 
# if __name__ == '__main__':
# 	loop = asyncio.get_event_loop()
# 	tasks = [
# 		asyncio.async(smart_fib(10)),
# 		asyncio.async(stupid_fib(10)),
# 	]
# 	loop.run_until_complete(asyncio.wait(tasks))
# 	print('All fib finished.')
# 	loop.close()	


async def smart_fib(n):
	index = 0
	a = 0
	b = 1
	while index < n:
		sleep_secs = random.uniform(0, 0.2)
		await asyncio.sleep(sleep_secs)
		print('Smart one think {} secs to get {}'.format(sleep_secs, b))
		a, b = b, a + b
		index += 1
 
async def stupid_fib(n):
	index = 0
	a = 0
	b = 1
	while index < n:
		sleep_secs = random.uniform(0, 0.4)
		await asyncio.sleep(sleep_secs)
		print('Stupid one think {} secs to get {}'.format(sleep_secs, b))
		a, b = b, a + b
		index += 1
 
if __name__ == '__main__':
	loop = asyncio.get_event_loop()
	tasks = [
		asyncio.ensure_future(smart_fib(10)),
		asyncio.ensure_future(stupid_fib(10)),
	]
	loop.run_until_complete(asyncio.wait(tasks))
	print('All fib finished.')
	loop.close()