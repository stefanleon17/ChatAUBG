#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <random>
#include <chrono>
#include <regex>

using namespace std;

const string MAP_FILE = "aubg_map.csv";
unordered_map<string, vector<string>> website_map;
unordered_map<string, int> website_length;		//stores the word/pair number of each particular website in storage; used for training only
const int WEBSITES_NUMBER = 250;				// number of websites considered; corresponds to the number of classes for the classification algorithms
const int MIN_REPLY = 100;						//minimum character length of a chatbot reply 
vector<string> redundant_words;					// stores stop words
const unsigned int SEED = 2931092510;


//Function for building dataset
void get_dataframe(const string& filename, unordered_map<string, vector<string>>& hashmap) {
	ifstream file(filename);
	string line;

	string current_key;
	vector<string> current_data;

	while (getline(file, line)) {
		//Check whether the line is a new website 
		if (line.find("http") == 0) {
			//When encountering a new key, save the current key & data to hashmap
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

	//Add the last key-data pair to hashmap
	if (!current_key.empty()) {
		hashmap[current_key] = current_data;
	}
}

//Functions for pre-processing
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

string smoothe(const string& text, bool special = false)
{
	string processed_text = text;

	//String of characters to keep for training classification models, and for processing input
	string keep = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890 ";

	//String of characters to keep for word generation purposes
	string special_keep = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890 .,-:;()!?";

	if (special == false)
	{
		for (int i = 0; i < processed_text.length(); i++)
			if (keep.find(processed_text[i]) == string::npos)
			{
				processed_text.erase(i, 1);
				i--;
			}
	}
	else 
	{
		for (int i = 0; i < processed_text.length(); i++)
			if (special_keep.find(processed_text[i]) == string::npos)
			{
				processed_text.erase(i, 1);
				i--;
			}
	}

	return processed_text;
}

//Additional structures constructed
struct Probabilities{
public:
	string entry_text;

	unordered_map<string, double> entry_count;

	Probabilities() {
		this->build_map();
		this->entry_text = "";
	}
private:
	void build_map() {
		for (auto iter = website_map.begin(); iter != website_map.end(); iter++)
			this->entry_count[iter->first] = 0.0001;												//applying Laplacian smoothing for later Naive Bayes processing
	}
};
vector <Probabilities> bigram;
vector <Probabilities> unigram;

struct MarkovChain {
public:
	string word;
	unordered_map<string, double> transitions;
	int total_entries;

	MarkovChain() {
		this->word = "";
		this->total_entries = 0;
	}
};
vector <MarkovChain> markov_chain;

//Functions for chatbot rule-setting
void eliminate_redundancies(vector<string>& input)
{
	string word = "";



	if (redundant_words.size() < 1)
	{
		ifstream file("stop_words.txt");
		string stop_word = "";

		while (!file.eof())
		{
			getline(file, stop_word, ',');
			redundant_words.push_back(stop_word);
		}
	}

	for (int i = 0; i < input.size(); i++)
	{
		word = input[i];
		to_lowercase(word);
		if (find(redundant_words.begin(), redundant_words.end(), word) != redundant_words.end())
		{
			input.erase(input.begin() + i);
			i--;
		}
	}
}

bool validate_input(const string& input)
{
	string correct = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890 .,-:;()!?/[]{}*&%$!";
	string check = "";
	for (char c : input)
		check += correct[correct.find(c)];

	if (check.find("???") != string::npos)
		return false;

	return true;
}


//Training functions
void train_markov_chain() {

	vector<string> content;
	MarkovChain* prev_entry = new MarkovChain();

	for (auto iter = website_map.begin(); iter != website_map.end(); iter++)
		for (int j = 0; j < iter->second.size(); j++)
		{
			if (iter->second[j].find(".") == string::npos)
				continue;

			string to_process = smoothe(iter->second[j], true);
			content = tokenize(to_process);

			if (content.size() < 4)
				continue;

			string prev_word = content[0] + " " + content[1] + " " + content[2];

			//checking whether we already have an entry for that word in our vector
			for (int i = 3; i < content.size(); i++) {

				bool exists = false;
				int index;

				for (index = 0; index < markov_chain.size(); index++)
					if (markov_chain[index].word == prev_word)
					{
						exists = true;
						break;
					}

				if (exists)
				{
					prev_entry = &markov_chain[index];
					prev_entry->word = prev_word;
					prev_entry->total_entries += 1;
				}
				else
				{
					MarkovChain empty_entry;
					markov_chain.push_back(empty_entry);
					prev_entry = &markov_chain[markov_chain.size() - 1];
					prev_entry->word = prev_word;
					prev_entry->total_entries = 1;
				}

				string next_word = content[i];
				exists = false;

				for (auto iter = prev_entry->transitions.begin(); iter != prev_entry->transitions.end(); iter++)
					if (iter->first == next_word)
					{
						exists = true;
						iter->second += 1;
						break;
					}
				if (!exists)
				{
					prev_entry->transitions[next_word] = 1;
				}
				prev_word = prev_word.substr(prev_word.find(" ") + 1) + " " + next_word;
			}
		}

	for (int i = 0; i < markov_chain.size(); i++)
		for (auto iter = markov_chain[i].transitions.begin(); iter != markov_chain[i].transitions.end(); iter++)
			iter->second = iter->second / markov_chain[i].total_entries;
}

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

			 eliminate_redundancies(content);

			 if (content.size() < 2)
				 continue;

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
	 {
		 double count = 0;
		 for (auto iter = bigram[i].entry_count.begin(); iter != bigram[i].entry_count.end(); iter++)
			 count += iter->second;

		 if (count < 2)
		 {
			 bigram.erase(bigram.begin() + i);
			 i--;
		 }

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

			 eliminate_redundancies(content);

			 if (content.empty())
				 continue;

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
	 {
		 double count = 0;
		 for (auto iter = unigram[i].entry_count.begin(); iter != unigram[i].entry_count.end(); iter++)
			 count += iter->second;
		 if (count < 2)
		 {
			 unigram.erase(unigram.begin() + i);
			 i--;
		 }
	 }

	 for (int i = 0; i < unigram.size(); i++)
		 calculate_probabilities(unigram[i]);
}

int train_models()
{	
	ofstream unigram_file("unigram.txt");
	ofstream bigram_file("bigram.txt");
	ofstream markov_file("markov_chain.txt");

	if (!unigram_file.is_open()) {
		cout << "Error: Unable to open unigram file for writing." << endl;
		return 1;
	}

	if (!bigram_file.is_open()) {
		cout << "Error: Unable to open bigram file for writing." << endl;
		return 2;
	}

	if (!markov_file.is_open()) {
		cout << "Error: Unable to open Markov Chain file for writing." << endl;
		return 1;
	}

	cout << endl << "Writing files successfully opened" << endl;

	train_bigram_model();

	cout << endl << "Finished training bigram model" << endl;

	for (const auto& entry : bigram) {
		bigram_file << entry.entry_text << endl;
		for (const auto& count : entry.entry_count) {
			bigram_file << count.first << ": " << count.second << endl;
		}
		bigram_file << endl;
	}

	cout << endl << "Finished writing bigram model" << endl;


	train_unigram_model();

	cout << endl << "Finished training unigram model" << endl;


	for (const auto& entry : unigram) {
		unigram_file << entry.entry_text << endl;
		for (const auto& count : entry.entry_count) {
			unigram_file << count.first << ": " << count.second << endl;
		}
		unigram_file << endl;
	}

	train_markov_chain();

	cout << "Finished training Markov model" << endl;

	for (int i = 0; i < markov_chain.size(); i++)
	{
		markov_file << markov_chain[i].word << endl;
		for (auto iter = markov_chain[i].transitions.begin(); iter != markov_chain[i].transitions.end(); iter++)
			markov_file << iter->first << ": " << iter->second << endl;
		markov_file << endl;
	}

	cout << endl << "Finished writing Markov model" << endl;

	bigram_file.close();
	unigram_file.close();
}
//Loader functions
void load_model(string source, vector<Probabilities>& target)			//Use smart_load_model,unless there is a specific need to load the entire model

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
			string key = "";
			double prob = 0;

			key = line.substr(0, split);
			prob = stod(line.substr(split + 2));

			entry.entry_count[key] = prob;
		}
		target.push_back(entry);
		getline(file, line);
	}

	file.close();
}

void load_markov_chain()
{
	ifstream file("markov_chain.txt");
	string line;

	while (!file.eof())
	{
		MarkovChain entry;

		getline(file, line);
		entry.word = line;

		getline(file, line);

		while (line != "")
		{
			size_t split = line.find(": ");
			string next_word = line.substr(0, split);
			double prob = stod(line.substr(split + 2));

			entry.transitions[next_word] = prob;
			getline(file, line);
		}
		markov_chain.push_back(entry);
	}
}

void create_index(string source_file, string target_file)
{
	ifstream source(source_file);
	ofstream target(target_file);

	int iterator = 0;
	string line;

	while (!source.eof())
	{
		getline(source, line);
		target << line << "," << iterator << endl;
		iterator++;

		for (int i = 1; i <= WEBSITES_NUMBER + 1; i++)
		{
			getline(source, line);
			iterator++;
		}
	}

	source.close();
	target.close();
}

Probabilities smart_load_model(unordered_map<string, int>& smart_map, string filename, string index_file, string word = "")
{
	ifstream file(filename);
	ifstream index(index_file);
	string line;

	if (word == "")
	{
		getline(index, line);

		while (!index.eof()) {

			size_t split = line.find(",");
			string key = line.substr(0, split);
			int line_number = stoi(line.substr(split + 1));

			smart_map[key] = line_number;
			getline(index, line);

		}
		Probabilities entry;
		return entry;
	}
	else
	{
		bool exists = false;
		for (auto iter = smart_map.begin(); iter != smart_map.end(); iter++)
		{
			if (word == iter->first)
			{
				exists = true;

				for (int i = 1; i <= iter->second + 1; i++)
					getline(file, line);

				Probabilities entry;
				entry.entry_text = word;
				 
				for (int i = 1; i <= WEBSITES_NUMBER; i++)
				{
					getline(file, line);
					size_t split = line.find(": ");

					string key = line.substr(0, split);
					double probability = stod(line.substr(split + 2));

					entry.entry_count[key] = probability;
				}
				
				return entry;
			}
		}
		if (!exists)
		{
			Probabilities entry;

			entry.entry_text = word;
			
			return entry;
		}
	}
}

//Functions for generating replies
string generate_text(vector<MarkovChain>& markov_chain)
{
	vector<string*> start_words;
	string text = "";
	for (int i = 0; i < markov_chain.size(); i++)
	{
		if (isupper(markov_chain[i].word[0]) && markov_chain[i].word[markov_chain[i].word.size() - 1] != '.')
			start_words.push_back(&markov_chain[i].word);

	}

	unsigned int seed = SEED;
	//unsigned int seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine rng(seed);
	
	uniform_int_distribution<int> words(0, start_words.size() - 1);
	int select = words(rng);

	string current_word = *start_words[select];
	text = current_word;


	//Condition searches for a word that ends the sentence in the dataset, bypassing possible abbreviation confounds
	while (!(current_word[current_word.size() - 1] == '.' && current_word.size() > 2) || text.size() < MIN_REPLY)
	{
		int index;
		for (index = 0; index < markov_chain.size(); index++)
			if (markov_chain[index].word == current_word)
				break;
		if (index > markov_chain.size() - 1)
		{
			seed += 2;
			rng.seed(seed);
				uniform_int_distribution<int> words(0, start_words.size() - 1);
			select = words(rng);
			current_word = *start_words[select];
		}
		for (index = 0; index < markov_chain.size(); index++)
		if (markov_chain[index].word == current_word)
			break;
		uniform_real_distribution<double> transit(0.0, 1.0);
		double random = transit(rng);
			double probability = 0.0;
		for (auto& transition : markov_chain[index].transitions)
		{
			probability += transition.second;
			if (random < probability)
			{
				current_word = current_word.substr(current_word.find(" ") + 1) + " " + transition.first;
				text = text + " " + current_word.substr(current_word.find(" ", current_word.find(" ") + 1) + 1);
				break;
			}
		}
	}
	return text;
}

void bias_generation(const string& website, vector<MarkovChain> model)
{
	vector<string> vec = website_map[website];
	vector <MarkovChain> new_model;

	MarkovChain* prev_entry = new MarkovChain();
	vector<string> content;

	for (int i = 0; i < vec.size(); i++)
	{
		if (vec[i].find(".") == string::npos)
			continue;

		string to_process = smoothe(vec[i], true);
		content = tokenize(to_process);

		if (content.size() < 4)
			continue;

		string prev_word = content[0] + " " + content[1] + " " + content[2];

		//checking whether we already have an entry for that word in our vector
		for (int i = 3; i < content.size(); i++) {

			bool exists = false;
			int index;

			for (index = 0; index < new_model.size(); index++)
				if (new_model[index].word == prev_word)
				{
					exists = true;
					break;
				}

			if (exists)
			{
				prev_entry = &new_model[index];
				prev_entry->word = prev_word;
				prev_entry->total_entries += 1;
			}
			else
			{
				MarkovChain empty_entry;
				new_model.push_back(empty_entry);
				prev_entry = &new_model[new_model.size() - 1];
				prev_entry->word = prev_word;
				prev_entry->total_entries = 1;
			}

			string next_word = content[i];
			exists = false;

			for (auto iter = prev_entry->transitions.begin(); iter != prev_entry->transitions.end(); iter++)
				if (iter->first == next_word)
				{
					exists = true;
					iter->second += 1;
					break;
				}
			if (!exists)
			{
				prev_entry->transitions[next_word] = 1;
			}
			prev_word = prev_word.substr(prev_word.find(" ") + 1) + " " + next_word;
		}
	}
	for (int i = 0; i < new_model.size(); i++)
		for (auto iter = new_model[i].transitions.begin(); iter != new_model[i].transitions.end(); iter++)
			iter->second = iter->second / new_model[i].total_entries;

	for (int i = 0; i < new_model.size(); i++)
	{
		bool exists = false;
		int j;
		for (j = 0; j < model.size(); j++)
			if (new_model[i].word == model[j].word)
			{
				exists = true;
				break;
			}
		if (exists)
		{
			for (auto entry : model[j].transitions)
				entry.second = 0.05 * entry.second + 0.95 * new_model[i].transitions[entry.first];
		}
		else {
			continue;
		}
	}
}


//Functions for classifying input
Probabilities unigram_predict_website(vector<string> input, vector<Probabilities>& unigram_loaded)
{
	//Smart Load

	unordered_map<string, int> unigram_load;
	smart_load_model(unigram_load, "unigram.txt", "unigram_index.txt");

	//Normal Load
	//load_model("unigram.txt", unigram);
	//cout << "Loaded unigram model" << endl;

	Probabilities unigram_score;
	for (auto iter = unigram_score.entry_count.begin(); iter != unigram_score.entry_count.end(); iter++)
		iter->second = log(iter->second);

	for (int i = 0; i < input.size(); i++)
	{
		Probabilities prob;

		bool exists = false;
		for (int j = 0; j < unigram_loaded.size(); j++)
		{
			if (input[i] == unigram_loaded[j].entry_text)
			{
				prob = unigram_loaded[j];
				exists = true;
				break;
			}
		}

		if (!exists)
		{
			prob = smart_load_model(unigram_load, "unigram.txt", "unigram_index.txt", input[i]);
			unigram_loaded.push_back(prob);
		}

		for (auto iter = unigram_score.entry_count.begin(); iter != unigram_score.entry_count.end(); iter++)
			iter->second += log(prob.entry_count[iter->first]);
	}
	return unigram_score;
}

Probabilities bigram_predict_website(vector<string> input, vector<Probabilities>& bigram_loaded)
{
	//Smart Load
	unordered_map<string, int> bigram_load;
	smart_load_model(bigram_load, "bigram.txt", "bigram_index.txt");

	//Normal Load
	//load_model("bigram.txt", bigram);
	//cout << "Loaded bigram model" << endl;

	Probabilities bigram_score;
	for (auto iter = bigram_score.entry_count.begin(); iter != bigram_score.entry_count.end(); iter++)
		iter->second = log(iter->second);
	if (input.size() < 2)
		return bigram_score;

	for (int i = 0; i < input.size() - 1; i++)
	{
		string pair = input[i] + " " + input[i + 1];
		Probabilities prob;

		bool exists = false;
		for (int j = 0; j < bigram_loaded.size(); j++)
		{
			if (pair == bigram_loaded[j].entry_text)
			{
				exists = true;
				prob = bigram_loaded[j];
				break;
			}
		}

		if (!exists)
		{
			prob = smart_load_model(bigram_load, "bigram.txt", "bigram_index.txt", pair);
			bigram_loaded.push_back(prob);
		}

		for (auto iter = bigram_score.entry_count.begin(); iter != bigram_score.entry_count.end(); iter++)
			iter->second += log(prob.entry_count[iter->first]);
	}

	return bigram_score;
}

//Run function
void chat()
{
	cout << "ChatAUBG: Hello! How can I help you today?" << endl;
	cout << endl << "> ";

	string input;

	while (true)
	{
		getline(cin, input);
		if (input == "exit")
			break;

		if (!validate_input(input))
		{
			cout << "ChatAUBG: I cannot understand what you are trying to say. Please try again." << endl;
			cout << endl << "> ";
			continue;
		}

		load_markov_chain();

		input = smoothe(input);
		to_lowercase(input);
		vector<string> to_process = tokenize(input);

		eliminate_redundancies(to_process);

		Probabilities unigram_score = unigram_predict_website(to_process, unigram);
		Probabilities bigram_score = bigram_predict_website(to_process, bigram);

		double max = -1. * (DBL_MAX);
		string max_key = "";

		Probabilities score;
		for (auto iter = score.entry_count.begin(); iter != score.entry_count.end(); iter++)
			iter->second = 0.55 * bigram_score.entry_count[iter->first] + 0.45 * unigram_score.entry_count[iter->first];

		for (auto iter = score.entry_count.begin(); iter != score.entry_count.end(); iter++)
		{
			if (iter->second > max)
			{
				max = iter->second;
				max_key = iter->first;
			}
		}

		bias_generation(max_key, markov_chain);
		cout << endl << "ChatAUBG: " << generate_text(markov_chain) << endl;
		cout << endl << "> ";
	}
}

void demo()
{

	cout << "ChatAUBG: Hello! How can I help you today?" << endl;
	cout << endl << "> ";

	string input;

	while (true)
	{
		getline(cin, input);
		if (input == "exit")
			break;

		if (!validate_input(input))
		{
			cout << endl << "ChatAUBG: I cannot understand what you are trying to say. Please try again." << endl;
			cout << endl << "> ";
			continue;
		}

		load_markov_chain();

		input = smoothe(input);
		to_lowercase(input);
		vector<string> to_process = tokenize(input);

		if (to_process.size() == 0)
		for (int i = 0; i < to_process.size(); i++)
			cout << to_process[i] << endl << endl;

		eliminate_redundancies(to_process);

		bool match = false;
		string match_reply = "";
		unordered_map<string, string> replies = {
		{"expect", "Once the Admissions Office has received all required documents, the Admissions Committee will consider the student's application. The student will be officially notified of the Committee's decision in a confidential letter sent to the mailing address provided in the application form. "},
		{"admission", "All you need to do to be considered is finish your application by our Priority Admission deadline, January 15. Once you submit your documents, we will automatically review you for all eligible scholarships."},
		{"admissions", "All you need to do to be considered is finish your application by our Priority Admission deadline, January 15. Once you submit your documents, we will automatically review you for all eligible scholarships."}
		};

		for (auto entry : replies)
			if (std::find(to_process.begin(), to_process.end(), entry.first) != to_process.end())
			{
				match = true;
				match_reply = entry.second;
				break;
			}

		cout << endl;
		cout << "Computing unigram prediction score..." << endl;
		Probabilities unigram_score = unigram_predict_website(to_process, unigram);
		cout << "Computed unigram score." << endl;
		cout << "Computing bigram prediction score..." << endl;
		Probabilities bigram_score = bigram_predict_website(to_process, bigram);
		cout << "Computed bigram score." << endl;

		double max = -1. * (DBL_MAX);
		string max_key = "";

		Probabilities score;
		for (auto iter = score.entry_count.begin(); iter != score.entry_count.end(); iter++)
			iter->second = 0.55 * bigram_score.entry_count[iter->first] + 0.45 * unigram_score.entry_count[iter->first];

		for (auto iter = score.entry_count.begin(); iter != score.entry_count.end(); iter++)
		{
			if (iter->second > max)
			{
				max = iter->second;
				max_key = iter->first;
			}
		}
		cout << "Computed UBIS score. Predicted website location: " << max_key << endl;
		cout << "Biasing Markov Chain algorithm..." << endl;


		if (match) 
		{
			cout << "Finished modifying the Markov Chain algorithm." << endl;
			cout << endl << "ChatAUBG: ";
			cout << match_reply << endl;
			cout << endl << "> ";
		}
		else
		{
			bias_generation(max_key, markov_chain);
			cout << "Finished modifying the Markov Chain algorithm." << endl;
			cout << endl << "ChatAUBG: ";
			cout << generate_text(markov_chain) << endl;
			cout << endl << "> ";
		}
		
	}
}

int main()
{
	get_dataframe(MAP_FILE, website_map);

	demo();

	return 0;

}



	