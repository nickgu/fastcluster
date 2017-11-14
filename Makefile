


target=kmeans

all: $(target)
	echo

clean:
	rm $(target)

kmeans: kmeans.cc
	g++ kmeans.cc -o kmeans -I/home/users/gusimiu/mycode/cppdev/src/ -lpthread



