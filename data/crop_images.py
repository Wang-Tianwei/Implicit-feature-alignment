import os
import cv2

if not os.path.exists('./wordsnew'):
	os.mkdir('./wordsnew')

lasthead = ''
for line in open('./words.txt', 'r').readlines():
	lsplit = line.split(' ')
	x = int(lsplit[3])
	y = int(lsplit[4])
	w = int(lsplit[5])
	h = int(lsplit[6])
	head1 = lsplit[0].split('-')[0]
	head2 = '-'.join(lsplit[0].split('-')[:-2])
	if not os.path.exists('./wordsnew/%s' % head1):
		os.mkdir('./wordsnew/%s' % head1)
	if not os.path.exists('./wordsnew/%s/%s' % (head1, head2)):
		os.mkdir('./wordsnew/%s/%s' % (head1, head2))
	imagename = lsplit[0]

	if head2 != lasthead:
		lasthead = head2
		imagefull = cv2.imread('./forms/%s.png' % head2, 0) # forms: all full-page images
		imagepatch = imagefull[y:y+h, x:x+w]
		cv2.imwrite('./wordsnew/%s/%s/%s.png' % (head1, head2, imagename), imagepatch)
	else:
		imagepatch = imagefull[y:y+h, x:x+w]
		cv2.imwrite('./wordsnew/%s/%s/%s.png' % (head1, head2, imagename), imagepatch)


if not os.path.exists('./linesnew'):
	os.mkdir('./linesnew')

lasthead = ''
for line in open('./labels.txt', 'r').readlines():
	lsplit = line.split(' ')
	x = int(lsplit[3])
	y = int(lsplit[4])
	w = int(lsplit[5])
	h = int(lsplit[6])
	head1 = lsplit[0].split('-')[0]
	head2 = '-'.join(lsplit[0].split('-')[:-2])
	if not os.path.exists('./linesnew/%s' % head1):
		os.mkdir('./linesnew/%s' % head1)
	if not os.path.exists('./linesnew/%s/%s' % (head1, head2)):
		os.mkdir('./linesnew/%s/%s' % (head1, head2))
	imagename = lsplit[0]

	if head2 != lasthead:
		lasthead = head2
		imagefull = cv2.imread('./forms/%s.png' % head2, 0)
		imagepatch = imagefull[y:y+h, x:x+w]
		cv2.imwrite('./linesnew/%s/%s/%s.png' % (head1, head2, imagename), imagepatch)
	else:
		imagepatch = imagefull[y:y+h, x:x+w]
		cv2.imwrite('./linesnew/%s/%s/%s.png' % (head1, head2, imagename), imagepatch)
