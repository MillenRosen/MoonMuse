import os
import random
import miditoolkit
from miditoolkit.midi import containers
import numpy as np
from itertools import chain

##############################
# constants
##############################

max_bars = 128

DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 16

DEFAULT_SCALE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
MAJOR_KEY = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
MINOR_KEY = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])

quality_conversion_table = {
    #           1     2     3     4  5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus4(b7)':[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus4(b7,9)':[1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    '9':       [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj9':    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '7(#9)':   [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj6(9)': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6(9)': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'maj(9)':  [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min(9)':  [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'maj(11)': [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min(11)': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    '11':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    'maj9(11)':[1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min11':   [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    'maj13':   [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    'min13':   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    #'5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    }
IDX_TO_KEY = {
    9: 'A',
    10: 'A#',
    11: 'B',
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#'
}
KEY_TO_IDX = {
    v: k for k, v in IDX_TO_KEY.items()
}
quality_name_table = {
    'M': 'maj',
    'm': 'min',
    '+': 'aug',
    'o': 'dim',
    'sus4': 'sus4',
    'sus2': 'sus2',
    '7': '7',
    'M7': 'maj7',
    'm7': 'min7',
    'o7': 'dim7',
    '/o7': 'hdim7',
    'None': 'None'
}
majorDegree2roman = {
    0: 'I',
    1: 'I#',
    2: 'II',
    3: 'II#',
    4: 'III',
    5: 'IV',
    6: 'IV#',
    7: 'V',
    8: 'V#',
    9: 'VI',
    10: 'VI#',
    11: 'VII',
}
roman2majorDegree = {v: k for k, v in majorDegree2roman.items()}
roman2minorDegree = {
    'I': 0,
    'I#': 1,
    'II': 2,
    'II#': random.choice([2, 3]),
    'III': 3,
    'IV': 5,
    'IV#': 6,
    'V': 7,
    'V#': random.choice([7, 8]),
    'VI': 8,
    'VI#': 9,
    'VII': 10
}

##############################
# containers for conversion
##############################
class ConversionEvent(object):
    def __init__(self, event, is_full_event=False):
        if not is_full_event:
            if 'Note' in event:
                self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
            elif 'Chord' in event:
                self.name, self.value = event.split('_')[0], '_'.join(event.split('_')[1:])
            else:
                self.name, self.value = event.split('_')
        else:
            self.name, self.value = event['name'], event['value']

    def __repr__(self):
        return 'Event(name: {} | value: {})'.format(self.name, self.value)

class NoteEvent(object):
    def __init__(self, pitch, bar, position, duration, velocity, microtiming=None):
        self.pitch = pitch
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
        self.quantized_duration = duration
        self.velocity = velocity

        if microtiming is not None:
            self.start_tick += microtiming

    def set_microtiming(self, microtiming):
        self.start_tick += microtiming

    def set_velocity(self, velocity):
        self.velocity = velocity

    def __repr__(self):
        return 'Note(pitch = {}, duration = {}, start_tick = {})'.format(
            self.pitch, self.quantized_duration, self.start_tick
        )

class TempoEvent(object):
    def __init__(self, tempo, bar, position):
        self.tempo = tempo
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)

    def set_tempo(self, tempo):
        self.tempo = tempo

    def __repr__(self):
        return 'Tempo(tempo = {}, start_tick = {})'.format(
            self.tempo, self.start_tick
        )

class ChordEvent(object):
    def __init__(self, chord_val, bar, position):
        self.chord_val = chord_val
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)

##############################
# conversion functions
##############################
def event_to_midi(key, events, mode, output_midi_path=None, is_full_event=False,
                  return_tempos=False, enforce_tempo=False, enforce_tempo_evs=None, play_chords=False):
    events = [ConversionEvent(ev, is_full_event=is_full_event) for ev in events]
    # print (events[:20])

    keyname = key.split('_')[1].upper()
    print('[degug] keyname:', keyname)
    start = np.where(MAJOR_KEY == keyname)[0][0]
    scale_range = np.concatenate([MAJOR_KEY[start:], MAJOR_KEY[:start]], axis=0)

    # assert events[0].name == 'Bar'
    temp_notes = []
    temp_tempos = []
    temp_chords = []
    temp_root = None
    temp_quality = None

    cur_bar = -1
    cur_position = 0

    for i in range(len(events)):
        # print('[debug] event:', events[i])
        if events[i].name == 'Bar':
            cur_bar += 1
        elif events[i].name == 'Beat':
            cur_position = int(events[i].value)
            assert cur_position >= 0 and cur_position < DEFAULT_FRACTION
        #   print (cur_bar, cur_position)
        elif events[i].name == 'Tempo' and 'Conti' not in events[i].value:
            temp_tempos.append(TempoEvent(
                int(events[i].value), max(cur_bar, 0), cur_position
            ))
        elif 'Note_Pitch' in events[i].name:
            if mode == 'full' and \
                    (i + 1) < len(events) and 'Note_Duration' in events[i + 1].name and \
                    (i + 2) < len(events) and 'Note_Velocity' in events[i + 2].name:
                # check if the 3 events are of the same instrument
                temp_notes.append(
                    NoteEvent(
                        pitch=int(events[i].value),
                        bar=cur_bar, position=cur_position,
                        duration=int(events[i + 1].value), velocity=int(events[i + 2].value)
                    )
                )
            elif mode == 'skyline' and \
                    (i + 1) < len(events) and 'Note_Duration' in events[i + 1].name:
                temp_notes.append(
                    NoteEvent(
                        pitch=int(events[i].value),
                        bar=cur_bar, position=cur_position,
                        duration=int(events[i + 1].value), velocity=80
                    )
                )
        elif 'Chord' in events[i].name and 'Conti' not in events[i].value:
            # print('[debug] chord:', events[i].value)
            temp_chords.append(
                ChordEvent(events[i].value, cur_bar, cur_position)
            )
        elif events[i].name in ['EOS', 'PAD']:
            continue
    
    # print('[debug] temp_chords:', temp_chords)
    # print('[debug] temp_notes:', temp_notes)
    print('[debug] temp_tempos:', temp_tempos)
    print('# tempo changes:', len(temp_tempos), '| # notes:', len(temp_notes))
    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.instruments = [
        miditoolkit.Instrument(program=0, is_drum=False, name='Piano')
    ]

    for n in temp_notes:
        midi_obj.instruments[0].notes.append(
            miditoolkit.Note(int(n.velocity), n.pitch, int(n.start_tick), int(n.start_tick + n.quantized_duration))
        )

    if enforce_tempo is False:
        for t in temp_tempos:
            midi_obj.tempo_changes.append(
                miditoolkit.TempoChange(t.tempo, int(t.start_tick))
            )
    else:
        if enforce_tempo_evs is None:
            enforce_tempo_evs = temp_tempos[1]
        for t in enforce_tempo_evs:
            midi_obj.tempo_changes.append(
                miditoolkit.TempoChange(t.tempo, int(t.start_tick))
            )

    for c in temp_chords:
        if 'None' in c.chord_val:
            midi_obj.markers.append(
                miditoolkit.Marker('Chord-{}'.format(c.chord_val), int(c.start_tick))
            )
        else:
            chord_val = c.chord_val
            root = chord_val.split('_')[0]
            if keyname in MAJOR_KEY:
                root = roman2majorDegree[root]
            else:
                root = roman2minorDegree[root]
            quality = chord_val.split('_')[1]
            c.chord_val = scale_range[int(root)] + '_' + quality
            midi_obj.markers.append(
                miditoolkit.Marker('Chord-{}'.format(c.chord_val), int(c.start_tick))
            )
    for b in range(cur_bar):
        midi_obj.markers.append(
            miditoolkit.Marker('Bar-{}'.format(b + 1), int(DEFAULT_BAR_RESOL * b))
        )

    midi_obj.max_tick = max([note.end for note in midi_obj.instruments[0].notes])

    if play_chords:
        add_chords(midi_obj)

    if output_midi_path is not None:
        midi_obj.dump(output_midi_path)

    if not return_tempos:
        return midi_obj
    else:
        return midi_obj, temp_tempos

def add_chords(midi_obj):
    default_velocity = 63

    markers = [marker for marker in midi_obj.markers if 'Chord' in marker.text]
    prev_chord = None
    dedup_chords = []
    for m in markers:
        if m.text == 'Chord-None_None':
            continue
        if m.text != prev_chord:
            prev_chord = m.text
            dedup_chords.append(m)
    markers = dedup_chords

    midi_maps = [chord_to_midi(marker.text.split('-')[1]) for marker in markers]
    midi_obj.instruments.append(containers.Instrument(program=0, is_drum=False, name='Piano'))

    if not len(midi_maps) == 0:
        for midi_map, prev_marker, next_marker in zip(midi_maps, markers[:-1], markers[1:]):
            for midi_pitch in midi_map:
                midi_note = containers.Note(start=prev_marker.time, end=next_marker.time, pitch=midi_pitch,
                                            velocity=default_velocity)
                midi_obj.instruments[1].notes.append(midi_note)
        for midi_pitch in midi_maps[-1]:
            midi_note = containers.Note(start=markers[-1].time, end=midi_obj.max_tick, pitch=midi_pitch,
                                        velocity=default_velocity)
            midi_obj.instruments[1].notes.append(midi_note)

    return midi_obj

def chord_to_midi(chord):
    root, quality = chord.split('_')
    bass = root

    root_c = 60
    bass_c = 36
    root_pc = KEY_TO_IDX[root]
    if quality in quality_name_table:
        quality = quality_name_table[quality]
    chord_map = list(np.where(np.array(quality_conversion_table[quality]) == 1)[0])
    bass_pc = KEY_TO_IDX[bass]

    return [bass_c + bass_pc] + [root_c + root_pc + i for i in chord_map]

def degree2pitch(key, octave, roman):
    # major key
    if key in MAJOR_KEY:
        tonic = KEY_TO_IDX[key]
        pitch = octave * 12 + tonic + roman2majorDegree[roman]
    # minor key
    elif key in MINOR_KEY:
        tonic = KEY_TO_IDX[key.upper()]
        pitch = octave * 12 + tonic + roman2minorDegree[roman]
    else:
        raise NameError('Wrong key name {}.'.format(key))

    return pitch

def extract_midi_events_from_generation(key, events, relative_melody=False):
    if relative_melody:
        new_events = []
        keyname = key.split('_')[1]
        root = None
        quality = None
        for evs in events:
            if 'Note_Octave' in evs:
                octave = int(evs.split('_')[2])
            elif 'Note_Degree' in evs:
                roman = evs.split('_')[2]
                pitch = degree2pitch(keyname, octave, roman)
                pitch = max(21, pitch)
                pitch = min(108, pitch)
                if pitch < 21 or pitch > 108:
                    raise ValueError('Pitch value must be in (21, 108), but gets {}'.format(pitch))
                new_events.append('Note_Pitch_{}'.format(pitch))
            elif 'Chord_Root' in evs:
                if 'None' in evs or 'Conti' in evs:
                    new_events.append(evs)
                else:
                    root = evs.split('_')[-1]
                    if keyname in MAJOR_KEY:
                        root = roman2majorDegree[root]
                    else:
                        root = roman2minorDegree[root]
            elif 'Chord_Quality' in evs:
                if 'None' in evs or 'Conti' in evs:
                    new_events.append(evs)
                else:
                    quality = evs.split('_')[-1]
                new_events.append('Chord_Root_{}'.format(root))
                new_events.append('Chord_Quality_{}'.format(quality))
            # elif 'Chord_' in evs:
            #     if 'None' in evs or 'Conti' in evs:
            #         new_events.append(evs)
            #     else:
            #         root, quality = evs.split('_')[1], evs.split('_')[2]
            #         if keyname in MAJOR_KEY:
            #             root = roman2majorDegree[root]
            #         else:
            #             root = roman2minorDegree[root]
            #         new_events.append('Chord_{}_{}'.format(root, quality))
            else:
                new_events.append(evs)
        events = new_events

    melody_starts = np.where(np.array(events) == 'Track_Melody')[0].tolist()
    perf_starts = np.where(np.array(events) == 'Track_Performance')[0].tolist()

    midi_bars = []
    for st, ed in zip(perf_starts, melody_starts[1:] + [len(events)]):
        bar_midi_events = events[st + 1: ed]
        midi_bars.append(bar_midi_events)

    return midi_bars

def txt_to_event(input_event_path):
    with open(input_event_path, 'r') as f:
        # 读取所有行并去除换行符
        events = [line.strip() for line in f.readlines()]
    return events


def merge_chord_events(data_list):
    new_list = []
    i = 0
    n = len(data_list)
    
    while i < n:
        current = data_list[i]
        
        # 检查当前元素是否是Chord_Root且下一个元素是Chord_Quality
        if (current.startswith('Chord_Root_') and 
            i+1 < n and 
            data_list[i+1].startswith('Chord_Quality_')):
            
            # 提取root和quality
            root = current.split('_')[-1]
            quality = data_list[i+1].split('_')[-1]
            
            # 创建合并后的chord
            merged_chord = f'Chord_{root}_{quality}'
            new_list.append(merged_chord)
            
            # 跳过下一个元素，因为已经处理了
            i += 2
        else:
            new_list.append(current)
            i += 1
            
    return new_list


# 替换为txt文件路径
# filedir = 'H:\PyHome\music\workdir\EMO-Disentanger-main\generation\perf\generation\perf\perf'
filedir = r'generation\stage3_performance_conn'

files = os.listdir(filedir)
for file in files:
    if file.endswith('.txt'):
        filename = os.path.join(filedir, file)
        output_midi_path = os.path.join(filedir, file.replace('.txt', '.mid'))
        generated = txt_to_event(filename)
        key = generated[1]
        
        # print(generated) # 打印event

        generated = merge_chord_events(generated)
        generated = extract_midi_events_from_generation(key, generated, relative_melody=True)

        # print(generated[0])
        event_to_midi(
            key,
            list(chain(*generated[:max_bars])),
            mode='full',
            output_midi_path=output_midi_path
        )