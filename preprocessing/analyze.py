from preprocess import *
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    #original = pd.read_csv(ORIGIN)
    #original = load_omni()
    original = pd.read_csv('combined.csv')
    original = map_names(original)
    clean = clean_values(original.copy())
    original = original.drop('definitive_diagnosis', axis=1)
#    original = original.drop('Definitive Diagnosis', axis=1)
    clean = clean.drop('definitive_diagnosis', axis=1) 
#    clean = clean.drop('Definitive Diagnosis', axis=1) 

    # Plot the frequency with which each columns contains nulls
    null_frequency = original.drop('multiple_birth_rank', axis=1).isnull().mean()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=null_frequency.index, y=null_frequency.values)
    plt.title("Frequency of Null Values in Columns (Original)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.savefig("null_frequency.png")
    plt.savefig("null_frequency.eps", format='eps', dpi=1000)
    plt.close()
    print("Null Frequency done.")

    # Plot the number of null values per row
    nulls = original.isnull().sum(axis=1)
    plt.figure(figsize=(12, 8))
    sns.histplot(nulls)
    plt.title("Number of Null Values per Row (Original)")
    plt.tight_layout()
    # plt.savefig("nulls_per_row_original.png")
    plt.savefig("nulls_per_row_origina.eps", format='eps', dpi=1000)
    plt.close()
    print("Nulls per row done.")

    # Plot the number of trimmed nulls per column
    nulls = clean.isnull().sum()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=nulls.index, y=nulls.values)
    plt.title("Cause of Filtered Rows by Column (Post Cleaning)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.savefig("nulls.png")
    plt.savefig("nulls.eps", format='eps', dpi=1000)
    plt.close()
    print("Nulls done cleaned.")

    # Plot the age distribution of the dataset comparing the original and cleaned
    plt.figure(figsize=(10, 6))
    sns.histplot(original["age_at_collection"].to_numpy(), color="red", label="Original")
    sns.histplot(clean["age_at_collection"].to_numpy(), color="blue", label="Cleaned")
    plt.title("Age Distribution") 
    plt.legend()
    plt.tight_layout()
    # plt.savefig("age_distribution.png")
    plt.savefig("age_distribution.eps", format='eps', dpi=1000)
    plt.close()
    print("Age distribution done.")

    # Plot the number of null values per row
    nulls = clean.isnull().sum(axis=1)
    plt.figure(figsize=(12, 8))
    sns.histplot(nulls)
    plt.title("Number of Null Values per Row (Post Cleaning)")
    plt.tight_layout()
    # plt.savefig("nulls_per_row.png")
    plt.savefig("nulls_per_row.eps", format='eps', dpi=1000)
    plt.close()
    print("Nulls per row done.")    



if __name__ == "__main__":
    main()
