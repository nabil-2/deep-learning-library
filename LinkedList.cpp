#include "LinkedList.h"

template<typename T>
LinkedList<T>::LinkedList() {

}

template<typename T>
LinkedList<T>::~LinkedList() {
}

template<typename T>
void LinkedList<T>::add(T element) {
	list.push_back(&T)
}

template<typename T>
void LinkedList<T>::del(int i) {
	*T adress = get(i);

}

template<typename T>
T* LinkedList<T>::get(int i) {
	return list.at(i);
}
