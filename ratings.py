import os
import pandas as pd
import pygame

def initialize_excel(file_path, audio_files):
    if not os.path.exists(file_path):
        columns = ['File Name', 'Aileen_Breathiness', 'Aileen_Pitch', 'Aileen_Smoothness', 'Aileen_Tone',
                  'Elijah_Breathiness', 'Elijah_Pitch', 'Elijah_Smoothness', 'Elijah_Tone',
                  'Isaiah_Breathiness', 'Isaiah_Pitch', 'Isaiah_Smoothness', 'Isaiah_Tone',
                  'Average_Breathiness', 'Average_Pitch', 'Average_Smoothness', 'Average_Tone']
        data = [[file] + [None] * 16 for file in audio_files]
        df = pd.DataFrame(data, columns=columns)
        df.to_excel(file_path, index=False)


def load_audio_files(directory):
    return [file for file in os.listdir(directory) if file.lower().endswith('.mp3')]


def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def stop_audio():
    pygame.mixer.music.stop()


def get_user_ratings():
    criteria = ['Breathiness', 'Pitch', 'Smoothness', 'Tone']
    ratings = []
    for criterion in criteria:
        while True:
            rating = input(f'Rate {criterion} (1-5): ').lower()
            if rating == 'quit':
                raise KeyboardInterrupt
            elif rating == '':
                pygame.mixer.music.play()
            else:
                try:
                    rating = float(rating)
                    if 1 <= rating <= 5:
                        ratings.append(rating)
                        break
                    else:
                        print('Invalid input. Please enter a number between 1 and 5.')
                except:
                    print('Invalid input. Please enter a number between 1 and 5.')

    return ratings


def update_averages(df, idx):
    for criterion in ['Breathiness', 'Pitch', 'Smoothness', 'Tone']:
        values = [df.at[idx, f'{user}_{criterion}'] for user in ['Aileen', 'Elijah', 'Isaiah'] if pd.notna(df.at[idx, f'{user}_{criterion}'])]
        if values:
            avg = sum(values) / len(values)
            df.at[idx, f'Average_{criterion}'] = round(avg, 2)


def main():
    try:
        directory = input('Enter the directory containing mp3 files: ')
        audio_files = load_audio_files(directory)
        excel_path = os.path.join(os.getcwd(), 'audio_ratings.xlsx')
        initialize_excel(excel_path, audio_files)

        user = input('Enter your name (Aileen, Elijah, Isaiah): ').capitalize()
        if user not in ['Aileen', 'Elijah', 'Isaiah']:
            print('Invalid user. Exiting.')
            return

        print("Type 'quit' at any time to exit and save your progress.")

        df = pd.read_excel(excel_path)
        def is_unrated(row):
            return pd.isna(row[f'{user}_Breathiness']) or str(row[f'{user}_Breathiness']).strip() == ''

        start_index = next((i for i, row in df.iterrows() if is_unrated(row)), None)

        if start_index is None:
            print('All files have been rated. Exiting.')
            return

        for idx, row in df.iloc[start_index:].iterrows():
            file_path = os.path.join(directory, row['File Name'])
            print(f'Playing: {row["File Name"]}')
            play_audio(file_path)

            ratings = get_user_ratings()
            for i, criterion in enumerate(['Breathiness', 'Pitch', 'Smoothness', 'Tone']):
                df.at[idx, f'{user}_{criterion}'] = ratings[i]
            update_averages(df, idx)
            stop_audio()

        df.to_excel(excel_path, index=False)
        print('All files rated.')

    except KeyboardInterrupt:
        stop_audio()
        df.to_excel(excel_path, index=False)
        print('\nProgress saved. Exiting.')


if __name__ == '__main__':
    main()
