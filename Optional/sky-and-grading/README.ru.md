# Дополнительные небеса и цветокоррекция

Этот комплект без игровых ресурсов собирает MPQ локально. Он добавляет небеса
зон для Burning Steppes, Blasted Lands и Mount Hyjal, собственную
цветокоррекцию benilla для Elwynn Forest, Duskwood, Westfall и Redridge, а с
ключом `--wf-colours` — умеренно смешанные ночные цвета из вашей установки WoW
Forever.

В репозитории находятся только скрипты и созданные проектом LUT/таблица. Модели,
текстуры и исходные DBC Blizzard читаются из ваших установок и записываются
только во внешний каталог сборки.

## Требования

- Windows и Python 3.8 или новее.
- Клиент World of Warcraft 1.12.1 и его каталог `Data`.
- Для небес — локальная установка WoW Forever `_classic_beta_` либо retail.
- Git и внешний `wowdev/pywowlib` точной ревизии ниже.
- 64-битный StormLib 9.40 (`StormLib.dll`) из официальных
  [релизов StormLib](https://github.com/ladislav-zezula/StormLib/releases/tag/v9.40).
  Скачайте `stormlib_dll.zip` и используйте x64 DLL.
- Интернет при первом запуске для публичного listfile и списка публичных TACT
  ключей, либо локальные файлы через `--listfile` и `--tact-keys`.

## Установка — одна команда на шаг

Команды выполняются в PowerShell; замените пути на свои.

1. Клонируйте pywowlib.

   ```powershell
   git clone https://github.com/wowdev/pywowlib.git C:\Tools\pywowlib
   ```

2. Выберите проверенную ревизию.

   ```powershell
   git -C C:\Tools\pywowlib checkout 55276dc5c2195da7fe136638a2a59716622f8c65
   ```

3. Установите зависимости Python. Сам pywowlib остаётся внешним checkout.

   ```powershell
   py -3 -m pip install bidict==0.23.1 multimethod numpy
   ```

4. Скачайте `stormlib_dll.zip` версии 9.40 по ссылке выше и распакуйте x64
   `StormLib.dll` в `C:\Tools\StormLib\StormLib.dll`.

5. Закройте игру и соберите кандидат. Скрипт не пишет в исходные установки.

   ```powershell
   py -3 .\tools\build_sky_patch.py --wow-forever "C:\Games\World of Warcraft" --client-data "C:\Games\WoW-1.12.1\Data" --pywowlib "C:\Tools\pywowlib" --stormlib "C:\Tools\StormLib\StormLib.dll" --out "C:\Temp\benilla-sky"
   ```

6. Сначала проверьте изолированную копию `Data`. Все файлы можно сделать
   жёсткими ссылками, но `patch-Z.mpq` должен быть обычной копией; замените его
   собранным кандидатом и выполните:

   ```powershell
   py -3 .\tools\verify_client_chain.py "C:\Temp\WoW-test\Data" --stage "C:\Temp\benilla-sky\stage" --stormlib "C:\Tools\StormLib\StormLib.dll"
   ```

7. Если живой `patch-Z.mpq` уже есть, создайте резервную копию.

   ```powershell
   Copy-Item "C:\Games\WoW-1.12.1\Data\patch-Z.mpq" "C:\Games\WoW-1.12.1\Data\patch-Z.before-benilla.mpq.bak"
   ```

8. Установите проверенный кандидат.

   ```powershell
   Copy-Item -Force "C:\Temp\benilla-sky\patch-Z.MPQ" "C:\Games\WoW-1.12.1\Data\patch-Z.mpq"
   ```

Если patch-Z раньше не существовал, пропустите шаг 7. Поле `archive_base` в
`build-report.json` показывает, был ли старый архив сохранён внутри кандидата.

### Варианты

Только собственная цветокоррекция, без современной установки и pywowlib:

```powershell
py -3 .\tools\build_sky_patch.py --grading-only --client-data "C:\Games\WoW-1.12.1\Data" --stormlib "C:\Tools\StormLib\StormLib.dll" --out "C:\Temp\benilla-grading"
```

Небеса и умеренные ночные цвета WoW Forever, извлечённые из вашей LightData:

```powershell
py -3 .\tools\build_sky_patch.py --wow-forever "C:\Games\World of Warcraft" --client-data "C:\Games\WoW-1.12.1\Data" --pywowlib "C:\Tools\pywowlib" --stormlib "C:\Tools\StormLib\StormLib.dll" --wf-colours --out "C:\Temp\benilla-sky-wf"
```

## Включение в игре

Включите **Zone Skyboxes** и **Colour Grading** в **Options → Advanced
Graphics**. Те же настройки через консоль:

```text
/console zoneSkyboxes 1
/console colorGrading 1
```

Значение `0` отключает эффект без удаления MPQ.

## Порядок загрузки MPQ

Все пять связанных DBC освещения должны загружаться из одного самого старшего
буквенного патча, который содержит любой из них. По умолчанию это
`patch-Z.mpq`. Если он уже существует, сборщик копирует его и накладывает новые
файлы, сохраняя постороннее содержимое; результат устанавливается вместо
старого архива.

Не переименовывайте результат вслепую. Более поздний архив с `Light.dbc`,
`LightParams.dbc`, `LightIntBand.dbc`, `LightFloatBand.dbc` или
`LightSkybox.dbc` разделит согласованный набор. Некоторые кастомные клиенты
используют символы, сортирующиеся после букв. Тогда объедините `stage` с
реально последним патчем либо удалите конфликтующие DBC и проверьте цепочку с
`--require-winner ANY`. Для обычного patch-Z оставьте строгую проверку.

Расширенный четырёхполевой `LightSkybox.dbc` предназначен для benilla. Не
используйте его с неизменённым оригинальным exe 1.12, ожидающим два поля.

## Удаление

Закройте игру. Если patch-Z существовал ранее, восстановите резервную копию:

```powershell
Move-Item -Force "C:\Games\WoW-1.12.1\Data\patch-Z.before-benilla.mpq.bak" "C:\Games\WoW-1.12.1\Data\patch-Z.mpq"
```

Если архив был создан этим комплектом с нуля, удалите только его:

```powershell
Remove-Item -LiteralPath "C:\Games\WoW-1.12.1\Data\patch-Z.mpq"
```

Удаление старого patch-Z уничтожит другое пользовательское содержимое.
Проверяйте `archive_base` и сохраняйте резервную копию.

## Решение проблем

- **Неверная ревизия pywowlib:** повторите шаг 2; сборщик намеренно не принимает
  другую ревизию.
- **StormLib не найден:** укажите `--stormlib` либо `STORMLIB_PATH`; DLL должна
  быть 64-битной для 64-битного Python.
- **CASC build/config не найден:** укажите корень World of Warcraft с
  `.build.info` и `Data` либо его дочерний `_classic_beta_`.
- **Зашифрованный/отсутствующий блок:** обновите публичный список TACT ключей.
  Приватные ключи не используются.
- **Неожиданная схема или номер клона:** другой аддон уже изменил выигрывающие
  DBC. Уберите конфликт или выполните осознанное объединение.
- **Небо не видно:** включите `zoneSkyboxes`, полностью перезапустите клиент и
  запустите `verify_client_chain.py` на его настоящем `Data`.
- **Ошибка band rows:** запустите `check_bands.py <Data> --stormlib <DLL>`.
  Правильные ключи: `(P-1)*18+b+1` и `(P-1)*6+b+1`.
- **Выходной архив уже существует:** выберите новый `--out` либо используйте
  `--force` только для намеренной замены кандидата.

## Авторы и права

Графика небес WoW Forever/retail принадлежит © Blizzard Entertainment и
извлекается пользователем локально из собственной установки. Репозиторий не
распространяет графику Blizzard или DBC клиента.

Метод цветокоррекции основан на `wxl-retail-grading` проекта WarcraftXL автора
iThorgrim и используется с указанием авторства. LUT и `MonkeyZoneGrade.dbc` в
`data/` созданы проектом benilla и лицензированы MIT OR Apache-2.0.
