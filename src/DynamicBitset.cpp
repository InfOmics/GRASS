#include "DynamicBitset.hpp"

/*
 * Class constructor - initialization of dataset
 */
DynamicBitset::DynamicBitset(unsigned int size) : size(size)
{
	this->words = ceil((float) size / ((float) sizeof(dataset_t) * (float) BIT_PER_DATASET_TYPE));
	this->dataset = new dataset_t[this->words];
	
	this->reset();
}

/*
 * Class destructor - free of dataset
 */
DynamicBitset::~DynamicBitset()
{
	delete this->dataset;
}

/*
 * Reset dataset
 */
void DynamicBitset::reset()
{
	memset(this->dataset, 0, this->words * sizeof(dataset_t));
}

/*
 * Set all data set to '1'
 */
void DynamicBitset::allset()
{
	memset(this->dataset, 0xFF, this->words * sizeof(dataset_t));
}

/*
 * Set the "pos" bit to '1' if value is true or '0' if value is false
 */
void DynamicBitset::set(unsigned int pos, bool value)
{
	int tpos = pos % BIT_PER_DATASET_TYPE;
	int dpos = pos / BIT_PER_DATASET_TYPE;
	
	dataset_t mask = value ? 128 : 127;
	
	if (value) {
		this->dataset[dpos] |= mask >> tpos;
	} else {
		for (int i = 0; i < tpos; i++) {
			dataset_t lastbit = 1;
			lastbit = mask & 1;
			mask = mask >> 1;
			mask |= lastbit ? 128 : 0;
		}
		
		this->dataset[dpos] &= mask;
	}
}

/*
 * Return true if the "pos" bit is '1'
 */
bool DynamicBitset::get(unsigned int pos)
{
	int tpos = pos % BIT_PER_DATASET_TYPE;
	int dpos = pos / BIT_PER_DATASET_TYPE;
	
	dataset_t mask = 128;
	mask = mask >> tpos;
	
	if (this->dataset[dpos] & mask)
		return true;
	else
		return false;
}

/*
 * Return number of '1' in dataset - length is the real length of dataset (without round of nearest 8bit)
 */
unsigned int DynamicBitset::onset(unsigned int length)
{
	unsigned int counter = 0;
	for (unsigned int i = 0; i < length; i++)
		if (this->get(i))
			counter++;
			
	return counter;
}

/*
 * Debug print - It prints starting from "left" side (set(0) is equal to 128)
 */
void DynamicBitset::print()
{
	/*cout << "Position: ";
	for (unsigned int i = 0; i < this->words * BIT_PER_DATASET_TYPE; i++)
		cout << i % 10;
		
	cout << endl << "Value:    ";*/
	
	for (unsigned int i = 0; i < this->words; i++) {
		
		dataset_t mask = 128, temp = 0;
		for (dataset_t j = 0; j < BIT_PER_DATASET_TYPE; j++) {			
			
			if (this->dataset[i] & mask)
				cout << "1";
			else
				cout << "0";				
			
			mask = mask >> 1;
		}
	}
	
	cout << endl;
}

/*
int main()
{
	DynamicBitset dynBitset(11);
	dynBitset.print();
	
	if (dynBitset.get(BIT_PER_DATASET_TYPE))
		cout << "is on" << endl;
	else
		cout << "is off" << endl;
	
	dynBitset.set(BIT_PER_DATASET_TYPE, true);
	dynBitset.set(9, true);
	dynBitset.set(10, true);
	dynBitset.set(0, true);
	dynBitset.set(1, false);
	
	if (dynBitset.get(BIT_PER_DATASET_TYPE))
		cout << "is on" << endl;
	else
		cout << "is off" << endl;
	
	dynBitset.print();
	
	dynBitset.set(BIT_PER_DATASET_TYPE, false);
	dynBitset.set(1, false);
	dynBitset.set(10, false);
	
	dynBitset.print();
	
	return 0;
}
*/
