{% if obj.display %}
	{% if obj.id != "pride" %}
		{% extends "python/module.rst" %}
	{% else %}
		{% extends "python/pride.rst" %}
	{% endif %}
{% else %}
	{% extends "python/module.rst" %}
{% endif %}
