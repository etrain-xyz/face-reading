function readURL(input, id) {
	if (input.files && input.files[0]) {
		var reader = new FileReader();
		reader.onload = function (e) {
			$('#' + id + '-preview').attr('src', e.target.result);
		}
		reader.readAsDataURL(input.files[0]); // convert to base64 string
	}
}

window.onload = function () {
	$(".image-input").change(function () {
		var id = $(this).attr('id')
		readURL(this, id);
		$("#loading-modal").removeClass('d-none');
		var input_face = document.querySelector('#face-input');
		var formData = new FormData();
		formData.append('face', input_face.files[0]);
		fetch("/detect", {
			method: 'post',
			body: formData,
		}).then(response => response.json()).then(data => {
			console.log(data)
			$("#loading-modal").addClass('d-none');
			if (data.face_desc && data.face_score) {
				var txt_color = "text-secondary";
				if (data.face_score >= 4)
					txt_color = "text-success";
				else if (data.face_score >= 3 && data.face_score < 4)
					txt_color = "text-primary";
				$("#face_rating").html('<p>Vẻ đẹp khuôn mặt:</p><p><span class="' + txt_color + '">' + data.face_score + '</span> / 5</p>');
				$("#face_cropped").attr('src', '/get_crop?image_name=' + data.face_desc.image_name);
				var html_content = '';
				for (var i = 0; i < data.face_desc.descriptions.length; i++) {
					var desc = data.face_desc.descriptions[i];
					html_content += `<tr><th scope="row">${desc.label}</th><td>${desc.analysis}</td></tr>`
				}
				$("#tbody-result").html(html_content);
				$('#result-dialog').modal('toggle');
			} else {
				if (data.message == "Many face")
					alert("Ảnh chính diện chỉ bao gồm một mặt người.")
				else
					alert("Ảnh chính diện quá nhỏ hoặc không đủ nét.")
			}
		});
	});

	$('#face-input-preview').click(function (e) {
		e.preventDefault();
		$('#face-input').trigger('click');
	})
}
