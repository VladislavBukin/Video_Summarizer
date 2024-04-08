import tkinter as tk
from youtube_transcript_api import YouTubeTranscriptApi

def get_video_subtitles(video_id):
    try:
        # Получаем доступные субтитры для видео
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Ищем первый непустой набор субтитров
        for transcript in transcript_list:
            if transcript.language_code:
                subtitles = transcript.fetch()
                if subtitles:
                    subtitles_text = ' '.join([sub['text'] for sub in subtitles])
                    return subtitles_text
        
        # Если не удалось найти ни один непустой набор субтитров
        return 'No subtitles found for the video.'

    except Exception as e:
        return f'An error occurred: {e}'

def on_submit():
    video_id = entry.get()
    subtitles = get_video_subtitles(video_id)
    output_window.config(state=tk.NORMAL)
    output_window.delete('1.0', tk.END)
    output_window.insert(tk.END, subtitles)
    output_window.config(state=tk.DISABLED)

root = tk.Tk()
root.title("YouTube Subtitles Extractor")

label = tk.Label(root, text="Enter YouTube Video ID:")
label.pack()

entry = tk.Entry(root)
entry.pack()

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack()

output_window = tk.Text(root, height=30, width=100, state=tk.DISABLED, font=("Terminus",18))
output_window.pack()

root.mainloop()
