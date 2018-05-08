#include<iostream>
using namespace std;

struct inflatable  // structure declaration
{
	char name[20];
	float volume;
	double price;
};
int main(){
	inflatable guest =
	{
		"Glorious Gloria",
		1.88,
		29.99
	};  // guest is a structure variable of type inflatable
	inflatable pal =
	{
		"Audacious Arthur",
		3.12,
		32.99
	};

	inflatable bouquet = { "sunflowers", 0.20, 12.49 };
	inflatable choice;
	cout << "bouquet: " << bouquet.name << " for $";
	cout << bouquet.price << endl;

	// initializing an array of structs
	inflatable guests[2] =
	{
		{ "Bambi",0.5,21.99 },
		{ "Godzilla",2000,565.99 }
	};

	cout << "The guests " << guests[0].name << " and " << guests[1].name
		<< "\nhave a combined volume of "
		<< guests[0].volume + guests[1].volume << " cubic feet." << endl;

	choice = bouquet; // assign one structure to another
	cout << "choice: " << choice.name << " for $";
	cout << choice.price << endl;

	cout << "Expand your guest list with " << guest.name;
	cout << " and " << pal.name << " !\n";
	cout << "You can have both for $";
	cout << guest.price + pal.price << " !\n";


	system("pause");
	return 0;
}