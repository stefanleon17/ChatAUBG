#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <algorithm>

using namespace std;

//string filename = "aubg_map.csv";
unordered_map<string, vector<string>> website_map;
unordered_map<string, int> website_length;
const int WEBSITES_NUMBER = 324;

//Functions for building dataset
void get_dataframe(const string& filename, unordered_map<string, vector<string>>& hashmap) {
	ifstream file(filename);
	string line;

	string current_key;
	vector<string> current_data;

	while (getline(file, line)) {
		// Check whether the line is a new website 
		if (line.find("http") == 0) {
			// When encountering a new key, save the current key & data to hashmap
			if (!current_key.empty()) {
				hashmap[current_key] = current_data;
				current_data.clear();
			}

			current_key = line;
		}
		else {
			if (line != "")
				current_data.push_back(line);
		}
	}

	// Add the last key-data pair to hashmap
	if (!current_key.empty()) {
		hashmap[current_key] = current_data;
	}
}


//functions for pre-processing
vector <string> tokenize (string& input)
{
	vector <string> words;
	istringstream in(input);

	string word;
	while (in>>word)
	{
		words.push_back(word);
	}
	
	return words;
}

void to_lowercase(string& text) 
{
	char* lowercase = new char[text.size() + 1];
	strcpy_s(lowercase,text.length() + 1, text.c_str());

	for (int i = 0; i < strlen(lowercase); i++) {
		lowercase[i] = tolower(lowercase[i]);
	}

	text = lowercase;

	delete[] lowercase;
}

string smoothe(const string& text)
{
	string processed_text = text;
	string remove = ".,/?!'-+=[]{}()*^";

	for (int i = 0; i < processed_text.length(); i++)
		if (remove.find(processed_text[i]) != string::npos)
			processed_text.erase(i,1);


	return processed_text;
}


struct Probabilities{
	string entry_text = "";

	unordered_map<string, double> entry_count;

	void build_map() {
		for (auto iter = website_map.begin(); iter != website_map.end(); iter++)
			entry_count[iter->first] = 1;		//applying Laplacian smoothing for later Naive Bayes processing
	}

	Probabilities() {
		build_map();
	}
};
vector <Probabilities> bigram;
vector <Probabilities> unigram;

//void calculate_class_length() {
//	vector <string> content;
//	for (auto iter = website_map.begin(); iter != website_map.end(); iter++) {
//		int length = 0;
//
//		for (int j = 0; j < iter->second.size(); j++) {
//
//			string to_process = smoothe(iter->second[j]);
//			to_lowercase(to_process);
//			content = tokenize(to_process);
//
//			for (int i = 0; i < content.size() - 1; i++) {
//
//				length++;
//			}
//		}
//		website_length[iter->first] = length;
//	}
//}

void calculate_probabilities(Probabilities& entry)
{
	for (auto iter = entry.entry_count.begin(); iter != entry.entry_count.end(); iter++)
		iter->second = iter->second / website_length[iter->first];
}

void train_bigram_model()
{
	 vector <string> content;

	 for (auto iter = website_map.begin(); iter != website_map.end(); iter++) {
		 int length = 0;
		 for (int j = 0; j < iter->second.size(); j++){ 
			 string to_process = smoothe(iter->second[j]);
			 to_lowercase(to_process);
			 content = tokenize(to_process);

			 for (int i = 0; i < content.size() - 1; i++) {
				 length++;

				 string entry_text = content[i] + " " + content[i + 1];
				 bool exists = false;
				 int index;

				 for (index = 0; index < bigram.size(); index++) {
					 if (entry_text == bigram[index].entry_text) {
						 exists = true;
						 break;
					 }
				 }
				 if (exists) 
					 bigram[index].entry_count[iter->first]++;

				 else {
					 Probabilities entry;
					 entry.entry_text = entry_text;

					 entry.entry_count[iter->first]++;

					 bigram.push_back(entry);
				 }
			 }
		 }
		 website_length[iter->first] = length;
	 }	
	 for (int i = 0; i < bigram.size(); i++)
		 calculate_probabilities(bigram[i]);
}



void train_unigram_model()
{
	vector <string> content;

	 for (auto iter = website_map.begin(); iter != website_map.end(); iter++) {
		 int length = 0;
		 for (int j = 0; j < iter->second.size(); j++){ 
			 string to_process = smoothe(iter->second[j]);
			 to_lowercase(to_process);
			 content = tokenize(to_process);

			 for (int i = 0; i < content.size(); i++) {
				 length++;

				 string entry_text = content[i];
				 bool exists = false;
				 int index;

				 for (index = 0; index < unigram.size(); index++) {
					 if (entry_text == unigram[index].entry_text) {
						 exists = true;
						 break;
					 }
				 }
				 if (exists) {
					 unigram[index].entry_count[iter->first]++;
				 }
				 else {
					 Probabilities entry;
					 entry.entry_text = entry_text;

					 entry.entry_count[iter->first] = 1;

					 unigram.push_back(entry);
				 }
			 }
		 }
		 website_length[iter->first] = length;
	 }	
	 for (int i = 0; i < unigram.size(); i++)
		 calculate_probabilities(unigram[i]);
}

void load_model(string source, vector<Probabilities>& target)
{
	ifstream file(source);
	string line;

	while (!file.eof())
	{
		Probabilities entry;

		getline(file, line);
		entry.entry_text = line;
		
		for (int i = 1; i <= WEBSITES_NUMBER; i++)
		{
			getline(file, line);
			size_t split = line.find(": ");

			string key = line.substr(0, split);
			double probability = atof(line.substr(split + 2).c_str());

			entry.entry_count[key] = probability;
		}
		target.push_back(entry);
		getline(file, line);
	}

	file.close();
}

Probabilities smart_load_model(unordered_map<string, int>& smart_map, string filename, string word = "")
{
	ifstream file(filename);
	string line;

	if (word == "")
	{
		while (!file.eof()) {
			int index = 0;
			getline(file, line);
			smart_map[line] = index;

			for (int i = 1; i <= WEBSITES_NUMBER + 1; i++)
				getline(file, line);
		}
		/*Probabilities entry;
		return entry;*/
	}
	else
	{
		bool exists = false;
		for (auto iter = smart_map.begin(); iter != smart_map.end(); iter++)
		{
			if (word == iter->first)
			{
				exists = true;

				for (int i = 1; i <= iter->second; i++)
					getline(file, line);

				Probabilities entry;
				entry.entry_text = word;

				for (int i = 1; i <= WEBSITES_NUMBER; i++)
				{
					getline(file, line);
					size_t split = line.find(": ");

					string key = line.substr(0, split);
					double probability = atof(line.substr(split + 2).c_str());

					entry.entry_count[key] = probability;
				}

				return entry;
			}
		}
		if (!exists)
		{
			Probabilities entry;
			for (auto iter = entry.entry_count.begin(); iter != entry.entry_count.end(); iter++)
				iter->second = iter->second / website_length[iter->first];
			return entry;
		}
	}
}


Probabilities bigram_predict_website(string& input)
{
	to_lowercase(input);
	vector <string> test = tokenize(input);
	Probabilities score;

	for (int i = 0; i < bigram.size(); i++)
		calculate_probabilities(bigram[i]);

	for (int i = 0; i < test.size() - 1; i++) {
		string text = test[i] + " " + test[i + 1];
		score.entry_text = text;

		for (int index = 0; index < bigram.size(); index++) {
			if (text == bigram[index].entry_text) {
				for (auto iter = bigram[index].entry_count.begin(); iter != bigram[index].entry_count.end(); iter++) {
					score.entry_count[iter->first] *= iter->second;
				}
				break;
			}
		}
	}

	return score;
}


Probabilities unigram_predict_website(string& input)
{
	to_lowercase(input);
	vector <string> test = tokenize(input);
	Probabilities score;

	for (int i = 0; i < test.size() - 1; i++) {
		string text = test[i];
		score.entry_text = text;

		for (int index = 0; index < unigram.size(); index++) {
			if (text == unigram[index].entry_text) {
				for (auto iter = unigram[index].entry_count.begin(); iter != unigram[index].entry_count.end(); iter++) {
					score.entry_count[iter->first] *= iter->second;
				}
				break;
			}
		}
	}

	return score;
}

//TODO
void train_markov_model() 
{

}
void focus_on_webpage(vector <Probabilities> markov_model) {}



int main()
{
	get_dataframe("aubg_map.csv", website_map);

	ofstream bigram_file("bigram.txt");

	if (bigram_file.is_open())
		cout << "Opened file" << endl;

	train_bigram_model();

	cout << endl << "Finished training bigram model" << endl;

	for (int i = 1; i < 10; i++) {
		cout << bigram[i].entry_text << endl;
		for (const auto& count : bigram[i].entry_count) {
			cout << count.first << ": " << count.second << endl;
		}
		cout << endl;
	}

	for (const auto& entry : bigram) {
		bigram_file << entry.entry_text << endl;
		for (const auto& count : entry.entry_count) {
			bigram_file << count.first << ": " << count.second << endl;
		}
		bigram_file << endl;
	}

	cout << endl << "Finished writing bigram model" << endl;

	/*unordered_map<string, int>

	cout << "Loading bigram..." << endl;
	smart_load_model()
	cout << "Bigram loaded successfully." << endl;
	cout << "Loading unigram..." << endl;
	load_model("unigram.txt", unigram);
	cout << "Unigram loaded successfully." << endl;

	for (int i = 0; i < 100; i++) {
		cout << "Entry Text: " << unigram[i].entry_text << endl;
		cout << "Entry Count:" << endl;
		for (const auto& entry : unigram[i].entry_count) {
			cout << entry.first << ": " << entry.second << endl;
		}
		cout << endl;
	}

	for (int i = 0; i < 100; i++) {
		cout << "Entry Text: " << unigram[i].entry_text << endl;
		cout << "Entry Count:" << endl;
		for (const auto& entry : unigram[i].entry_count) {
			cout << entry.first << ": " << entry.second << endl;
		}
		cout << endl;
	}*/

	return 0;

}



	/*for (int i = 0; i < bigram.size(); i++)
		for (auto iter = bigram[i].entry_count.begin(); iter != bigram[i].entry_count.end(); iter++)
			cout << iter->first << " " << iter->second << endl;

	string input = "I'm looking for information on the financial aid deadline";
	Probabilities result1 = bigram_predict_website(input);
	Probabilities result2 = unigram_predict_website(input);

	cout << endl << input << endl;
	
	for (auto iter = result1.entry_count.begin(); iter != result1.entry_count.end(); iter++)
		cout << iter->first << " " << iter->second << endl;
	cout << endl;

	for (auto iter = result2.entry_count.begin(); iter != result2.entry_count.end(); iter++)
		cout << iter->first << " " << iter->second << endl;
	cout << endl;*/
